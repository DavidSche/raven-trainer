"""Layer 3: Quality Check -- filter low-quality frames.

Checks:
  1. Blurry (Laplacian variance)
  2. Black frame (mean pixel value)
  3. Overexposed (mean + hot pixel ratio)
  4. Resolution (min size)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core import ImportedSample, IngestionReport, QualityFilterConfig


class QualityChecker:
    """Run quality checks on sampled frames."""

    def __init__(self, config: Optional[QualityFilterConfig] = None):
        self.config = config or QualityFilterConfig()
        self.last_report = IngestionReport()

    def check(self, samples: List[ImportedSample], output_dir: Path) -> List[ImportedSample]:
        """Filter samples that pass quality checks. Copy valid ones to output_dir."""
        kept: List[ImportedSample] = []
        report = IngestionReport()
        report.total_sampled = len(samples)

        for sample in samples:
            path = sample.frame_path
            if not path.exists():
                continue

            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            laplacian_var = compute_laplacian_variance(img)
            mean_brightness = compute_mean_brightness(img)

            # Check resolution
            if w < self.config.min_width or h < self.config.min_height:
                report.quality_removed_small += 1
                # M-36: do NOT unlink source frame here. The ingestion pipeline
                # should only filter; rejected files are cleaned up at the end
                # of the pipeline. Deleting here makes the pipeline non-idempotent
                # (a re-run would find no frames to inspect).
                continue

            # Check blur
            if laplacian_var < self.config.blur_laplacian_threshold:
                report.quality_removed_blurry += 1
                continue

            # Check black frame
            if mean_brightness < self.config.black_mean_threshold:
                report.quality_removed_black += 1
                continue

            # Check overexposed
            overexposed_ratio = float(np.mean(img > self.config.overexposed_mean_threshold))
            if mean_brightness > self.config.overexposed_mean_threshold or \
               overexposed_ratio > self.config.overexposed_pixel_ratio:
                report.quality_removed_overexposed += 1
                continue

            # Quality score (0-1, higher is better)
            quality = min(1.0, laplacian_var / 500.0)
            sample.quality_score = quality
            sample.width = w
            sample.height = h
            sample.metadata["laplacian_var"] = float(laplacian_var)
            sample.metadata["mean_brightness"] = float(mean_brightness)

            kept.append(sample)

        report.total_kept_after_quality = len(kept)
        self.last_report = report
        print(f"    Quality check: kept {len(kept)} / {len(samples)} "
              f"(blurry={report.quality_removed_blurry}, "
              f"black={report.quality_removed_black}, "
              f"overexposed={report.quality_removed_overexposed}, "
              f"small={report.quality_removed_small})")
        return kept


def compute_laplacian_variance(img: np.ndarray) -> float:
    """Compute Laplacian variance for blur detection.
    Higher values = sharper image. Threshold ~100 is typical.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_mean_brightness(img: np.ndarray) -> float:
    """Compute mean pixel value across all channels (0-255)."""
    return float(np.mean(img))

