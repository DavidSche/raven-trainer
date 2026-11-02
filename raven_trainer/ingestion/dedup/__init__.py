"""Layer 4: Deduplicate -- remove near-duplicate images.

Methods:
  - dHash: gradient-based difference hash (fast)
  - pHash: perceptual hash (more accurate, slower)
  - SSIM: structural similarity (most accurate, slowest)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from ..core import DedupConfig, ImportedSample, IngestionReport


class Deduplicator:
    """Remove near-duplicate frames using perceptual hashing."""

    def __init__(self, config: Optional[DedupConfig] = None):
        self.config = config or DedupConfig()

    def dedup(self, samples: List[ImportedSample], output_dir: Path) -> List[ImportedSample]:
        """Remove duplicates, return unique samples (no file copying needed)."""
        if not samples:
            return []

        report = IngestionReport()
        report.total_kept_after_quality = len(samples)

        # Compute hashes
        hash_fn = self._get_hash_fn()
        hash_map: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        # hash_map[hash_str] = [(sample_index, quality_score), ...]

        for i, sample in enumerate(samples):
            path = sample.frame_path
            if not path.exists():
                continue
            try:
                h = hash_fn(path)
                hash_map[str(h)].append((i, sample.quality_score))
            except (OSError, ValueError, TypeError) as exc:
                # M-37: catch specific exceptions instead of bare
                # `except Exception: pass` which silently swallowed all errors
                # (including corrupted images, permission issues, etc.)
                import sys
                sys.stderr.write(f"    [Dedup] Skipping {path}: {type(exc).__name__}: {exc}\n")

        # Dedup within each hash bucket (hamming distance)
        kept_indices: set = set()

        for h, candidates in hash_map.items():
            # Sort by quality descending, keep best
            candidates.sort(key=lambda x: -x[1])
            kept_idx = candidates[0][0]
            kept_indices.add(kept_idx)

            for idx, _ in candidates[1:]:
                # Check hamming distance
                h1 = imagehash.hex_to_hash(h)
                h2 = hash_fn(samples[idx].frame_path)
                if h1 - h2 <= self.config.hash_distance:
                    # Remove duplicate file
                    samples[idx].frame_path.unlink(missing_ok=True)
                    report.dedup_removed += 1
                else:
                    kept_indices.add(idx)

        # SSIM pass for remaining near-duplicates across buckets
        if self.config.method == "ssim":
            kept_list = sorted(kept_indices)
            final_kept = self._ssim_pass(samples, kept_list, report)
        else:
            final_kept = sorted(kept_indices)

        result = [samples[i] for i in final_kept]
        report.total_after_dedup = len(result)
        self.last_report = report

        print(f"    Dedup ({self.config.method}): kept {len(result)} / {len(samples)} "
              f"(removed {report.dedup_removed})")
        return result

    def _get_hash_fn(self):
        # M-16: imagehash.phash/dhash accept a path and internally call
        # Image.open() without closing it, leaking file descriptors when called
        # thousands of times. Open the image ourselves inside a `with` block so
        # the fd is released deterministically.
        hash_func = imagehash.phash if self.config.method == "phash" else imagehash.dhash

        def fn(p):
            with Image.open(p) as img:
                return hash_func(img, hash_size=self.config.hash_size)

        return fn

    def _ssim_pass(
        self,
        samples: List[ImportedSample],
        candidates: List[int],
        report: IngestionReport,
    ) -> List[int]:
        """Additional SSIM check on remaining candidates.

        M-22: Cache the resized grayscale image alongside its index in ``final``
        so the inner comparison loop reuses the in-memory array instead of
        re-running ``cv2.imread`` + ``cv2.resize`` for every pair (O(N²) reads).
        """
        # final entries: (sample_index, resized_grayscale_array)
        final: List[Tuple[int, "np.ndarray"]] = []
        final_indices: List[int] = []
        for idx in candidates:
            should_keep = True
            img_a = cv2.imread(str(samples[idx].frame_path), cv2.IMREAD_GRAYSCALE)
            if img_a is None:
                continue
            img_a = cv2.resize(img_a, (224, 224))

            for _, img_b in final:
                score = ssim(img_a, img_b)
                if score >= self.config.ssim_threshold:
                    should_keep = False
                    samples[idx].frame_path.unlink(missing_ok=True)
                    report.dedup_removed += 1
                    break

            if should_keep:
                final.append((idx, img_a))
                final_indices.append(idx)

        return final_indices
