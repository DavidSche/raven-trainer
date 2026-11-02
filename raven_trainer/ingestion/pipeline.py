"""Orchestrate ingestion pipeline with streaming sampling pipeline.

After ingestion completes, results are automatically registered
as a new version in the Data Asset Management system.
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

from ..config import R
from typing import Any, Dict, List, Optional

from .core import (
    ImportedSample, IngestionReport,
    QualityFilterConfig, DedupConfig,
)
from .importers import RawImporter
from .sampling import build_pipeline, run_pipeline
from .sampling.core import FrameMeta
from .quality import QualityChecker
from .dedup import Deduplicator
from .scene import SceneAnalyzer
from .report import ReportGenerator
from .sampling.core import DecodeConfig as _DecodeConfig


# Image file extensions recognised as valid frame sources
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class IngestionPipeline:
    """Full ingestion pipeline orchestrator with streaming sampling."""

    def __init__(
        self,
        raw_root: str | Path,
        sampling_pipeline: Optional[List[str]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        sample_count: int = 100,
        quality: Optional[QualityFilterConfig] = None,
        dedup: Optional[DedupConfig] = None,
        auto_register: bool = True,
        target_dataset: str = "",
        decode_config: Optional[_DecodeConfig] = None,
    ):
        self.raw_root = Path(raw_root)
        self.sampling_pipeline = sampling_pipeline or ["interval"]
        self.sampler_config = sampler_config or {}
        self.sample_count = sample_count
        self.auto_register = auto_register
        self.target_dataset = target_dataset  # explicit dataset name to register into
        self.decode_config = decode_config
        self.importer = RawImporter(raw_root)
        self.quality = QualityChecker(quality)
        self.dedup = Deduplicator(dedup)
        self.scene = SceneAnalyzer()
        self.report_gen = ReportGenerator()

    def run(
        self,
        output_dir: str | Path = "",
        skip_scene: bool = True,
        generate_report: bool = True,
    ) -> IngestionReport:
        t_start = _time.time()
        report = IngestionReport()
        out = Path(output_dir) if output_dir else R.INGESTION_RUNS

        print(f"[{_time.strftime('%H:%M:%S')}] Step 1/5: Scanning raw sources...")
        sources = self.importer.scan()
        report.total_sources = len(sources)
        print(f"  Found {len(sources)} source(s)")
        for s in sources:
            meta = s.metadata
            if meta:
                print("    {meta.source_name:25s} {meta.source_type:8s} {meta.resolution or '?':12s} {meta.total_frames or 0:>8} frames")

        if not sources:
            print("[Ingestion] No raw sources found.")
            return report

        pipeline_stages = build_pipeline(self.sampling_pipeline)
        for stage in pipeline_stages:
            name = stage.name
            if name in self.sampler_config:
                for k, v in self.sampler_config[name].items():
                    if hasattr(stage, k):
                        setattr(stage, k, v)

        all_kept: List[FrameMeta] = []
        step_name = " -> ".join(self.sampling_pipeline)
        print(f"[{_time.strftime('%H:%M:%S')}] Step 2/5: Sampling pipeline [{step_name}]")

        for src in sources:
            if not src.metadata:
                print(f"  Skipping unknown source: {src.path.name}")
                continue

            if src.metadata.source_type == "image_dir":
                # ── Image directory: load images directly, skip video sampling ──
                total_images = src.metadata.total_frames or 0
                if total_images == 0:
                    print(f"  [SKIP] Empty image directory: {src.path.name}")
                    continue

                print(f"\n  >>> Processing image directory: {src.path.name} ({total_images} images)")
                image_files = sorted(
                    [f for f in src.path.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                )

                # Sample subset if sample_count is set
                selected = image_files
                if self.sample_count > 0 and len(image_files) > self.sample_count:
                    step = max(1, len(image_files) // self.sample_count)
                    selected = image_files[::step][:self.sample_count]

                # Create output frame directory
                frame_dir = out / src.metadata.source_name
                frame_dir.mkdir(parents=True, exist_ok=True)

                saved = 0
                for img_path in selected:
                    dst = frame_dir / "{saved:06d}{img_path.suffix}"
                    try:
                        import shutil
                        shutil.copy2(str(img_path), str(dst))
                    except Exception as copy_err:
                        print(f"    [WARN] copy error for {img_path.name}: {copy_err}")
                        continue

                    frame_meta = FrameMeta(
                        source_name=src.metadata.source_name,
                        frame_idx=saved,
                        timestamp_sec=0.0,
                        saved_path=dst,
                        selected=True,
                    )
                    all_kept.append(frame_meta)
                    saved += 1

                src_name = src.metadata.source_name
                report.source_distribution[src_name] = saved
                print(f"  <<< Collected {saved} images from {src_name}")
                continue

            # ── Video source: use the sampling pipeline ──
            if src.metadata.source_type != "video":
                print(f"  Skipping unsupported source type: {src.path.name} ({src.metadata.source_type})")
                continue

            total_frames = src.metadata.total_frames or 0
            if total_frames == 0:
                print(f"  [SKIP] Unknown frame count: {src.path.name}")
                continue

            print(f"\n  >>> Processing: {src.path.name} ({total_frames} frames, {src.metadata.resolution})")

            try:
                kept = run_pipeline(
                    video_path=src.path,
                    output_dir=out,
                    pipeline=pipeline_stages,
                    fps=src.metadata.fps or 25.0,
                    total_frames=total_frames,
                    sample_count=self.sample_count,
                    decode_config=self.decode_config,
                )
                all_kept.extend(kept)
                src_name = src.metadata.source_name
                report.source_distribution[src_name] = len(kept)
                print(f"  <<< Pipeline output: {len(kept)} frames from {src_name}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        report.total_sampled = len(all_kept)
        report.total_raw_frames = report.total_sampled
        print(f"[{_time.strftime('%H:%M:%S')}] Sampling complete: {report.total_sampled} frames")

        print(f"[{_time.strftime('%H:%M:%S')}] Step 3/5: Quality check ({len(all_kept)} frames)...")
        quality_passed: List[ImportedSample] = []
        if all_kept:
            samples = [
                ImportedSample(
                    source_name=m.source_name or str(Path(m.saved_path).parent.name) if m.saved_path else "",
                    frame_path=m.saved_path,
                    frame_index=m.frame_idx,
                    timestamp_sec=m.timestamp_sec,
                    motion_score=m.motion_score,
                    quality_score=1.0,
                )
                for m in all_kept
                if m.saved_path
            ]
            quality_passed = self.quality.check(samples, out)
            report.total_kept_after_quality = len(quality_passed)
            if hasattr(self.quality, "last_report"):
                lr = self.quality.last_report
                report.quality_removed_blurry = lr.quality_removed_blurry
                report.quality_removed_black = lr.quality_removed_black
                report.quality_removed_overexposed = lr.quality_removed_overexposed
                report.quality_removed_small = lr.quality_removed_small
            print(f"  Quality check: kept {report.total_kept_after_quality}/{len(samples)} "
                  f"(blurry={report.quality_removed_blurry}, black={report.quality_removed_black}, "
                  f"overexposed={report.quality_removed_overexposed}, small={report.quality_removed_small})")

            if quality_passed:
                print(f"[{_time.strftime('%H:%M:%S')}] Step 4/5: Dedup ({len(quality_passed)} frames)...")
                dedup_passed = self.dedup.dedup(quality_passed, out)
                report.total_after_dedup = len(dedup_passed)
                report.dedup_removed = len(quality_passed) - len(dedup_passed)
                print(f"  Dedup: kept {report.total_after_dedup}, removed {report.dedup_removed}")
                final = dedup_passed
            else:
                final = []
        else:
            final = []

        report.samples = [str(s.frame_path) for s in final] if final else []

        # Drop rejected frames now that all stages have finished. Quality/Dedup
        # used to unlink mid-pipeline, which broke re-runs and left nothing to
        # inspect when a run failed halfway. Cleanup is scoped to this run's
        # output dir so a manifest can never point at a deleted file.
        self._cleanup_rejected(out, {s.frame_path for s in final})

        elapsed = _time.time() - t_start
        print("\n[{_time.strftime('%H:%M:%S')}] Step 5/5: Complete in {elapsed:.1f}s")
        print(f"  Final kept: {len(final)} frames")
        print(report.summary())

        if generate_report:
            print(f"[{_time.strftime('%H:%M:%S')}] Generating HTML report...")
            report_path = out / "ingestion_report.html"
            self.report_gen.generate(report, report_path)
            print(f"  Report saved: {report_path}")

        # JSON report
        json_path = out / "ingestion_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_sources": report.total_sources,
                "total_sampled": report.total_sampled,
                "total_kept_after_quality": report.total_kept_after_quality,
                "total_after_dedup": report.total_after_dedup,
                "quality_removed_blurry": report.quality_removed_blurry,
                "quality_removed_black": report.quality_removed_black,
                "quality_removed_overexposed": report.quality_removed_overexposed,
                "quality_removed_small": report.quality_removed_small,
                "dedup_removed": report.dedup_removed,
                "elapsed_seconds": round(elapsed, 1),
                "source_distribution": report.source_distribution,
            }, f, indent=2, ensure_ascii=False)

        # Auto-register as Data Asset version
        if self.auto_register and final and report.total_after_dedup > 0:
            self._auto_register_asset(final, out, report)

        return report

    @staticmethod
    def _cleanup_rejected(output_dir: Path, keep: set) -> None:
        """Remove frames in ``output_dir`` that no stage kept."""
        try:
            root = Path(output_dir).resolve()
            if not root.is_dir():
                return
            for entry in root.rglob("*"):
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                if entry in keep:
                    continue
                if not entry.resolve().is_relative_to(root):
                    continue
                entry.unlink(missing_ok=True)
        except OSError as exc:
            print(f"  [Cleanup] Skipped: {exc}")

    def _auto_register_asset(
        self,
        final_samples: List[ImportedSample],
        output_dir: Path,
        report: IngestionReport,
    ) -> None:
        """Automatically register ingestion results as a new Data Asset version.

        Frame files are archived to storage/images/<source_name>/ (immutable layer)
        and the manifest records relative paths from PROJECT_ROOT.
        """
        try:
            from ..data_asset import (
                AssetStore, DatasetRegistry,
                VersionManager, MetadataStore, ImageMetadata,
                LineageStore, LineageNode, LineageEdge,
            )
            from ..data_asset.version import VersionSnapshot
            import shutil as _shutil

            data_assets_root = R.DATASETS
            if not data_assets_root.exists():
                print("  [Auto-Register] data/datasets/ not found, skipping auto-register")
                return

            # ── 1. Determine dataset name ──
            # Priority: explicit target_dataset > raw_root name > first source name
            if self.target_dataset:
                dataset_name = self.target_dataset
            elif self.raw_root.name and self.raw_root.name not in ("raw", "storage", "."):
                dataset_name = self.raw_root.name
            else:
                source_names = list(report.source_distribution.keys())
                dataset_name = source_names[0] if source_names else "unnamed"

            registry = DatasetRegistry(AssetStore(data_assets_root))

            # ── 2. Create or get the dataset ──
            existing = registry.get(dataset_name)
            if existing is None:
                registry.create(
                    name=dataset_name,
                    task="detection",
                    display_name=dataset_name.replace("_", " ").title(),
                )
                existing = registry.get(dataset_name)
                print(f"  [Auto-Register] Created dataset '{dataset_name}'")

            # ── 3. Archive frames to storage/images/<source_name>/ (immutable) ──
            # Use dataset_name as the source sub-directory name
            storage_target = R.STORAGE_IMAGES / dataset_name
            storage_target.mkdir(parents=True, exist_ok=True)

            archived: List[Path] = []
            archived_source_names: List[str] = []
            for s in final_samples:
                if not s.frame_path or not s.frame_path.exists():
                    continue
                dst = storage_target / s.frame_path.name
                if dst.exists():
                    stem, suffix = s.frame_path.stem, s.frame_path.suffix
                    dst = storage_target / "{stem}_{len(archived):04d}{suffix}"
                try:
                    _shutil.copy2(str(s.frame_path), str(dst))
                    archived.append(dst)
                    archived_source_names.append(s.source_name)
                except Exception as copy_err:
                    print(f"  [Auto-Register] copy error: {copy_err}")

            print(f"  [Auto-Register] Archived {len(archived)} frames -> {storage_target}")

            if not archived:
                print("  [Auto-Register] No frames archived, skipping version creation")
                return

            # ── 4. Build manifest records with relative storage_path ──
            manifest_records = []
            for i, archived_path in enumerate(archived):
                try:
                    rel_path = archived_path.resolve().relative_to(R.PROJECT_ROOT)
                    storage_path = rel_path.as_posix()  # e.g. data/storage/images/workshop_a/frame_0001.jpg
                except ValueError:
                    storage_path = str(archived_path)  # fallback: absolute
                manifest_records.append({
                    "image_id": "{dataset_name}_{i:06d}",
                    "storage_path": storage_path,
                    "source": archived_source_names[i],
                    "capture_time": "",
                    "camera": archived_source_names[i],
                })

            # ── 5. Create version with manifest ──
            asset_dir = data_assets_root / dataset_name
            vm = VersionManager(asset_dir)

            existing_versions = vm.list_versions()
            version_str = vm.next_version_str()
            version_dir = vm.get_version_dir(version_str)

            # Clean broken version dir (exists without manifest -> previous failure)
            if version_dir.exists() and not (version_dir / "manifest.jsonl").exists():
                _shutil.rmtree(str(version_dir))

            version_dir.mkdir(parents=True, exist_ok=True)

            # Write manifest
            manifest_path = version_dir / "manifest.jsonl"
            with open(manifest_path, "w", encoding="utf-8") as mf:
                for rec in manifest_records:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Write snapshot
            snap = VersionSnapshot(
                version=version_str,
                parent_version=existing_versions[0].version if existing_versions else None,
                description=f"Auto-ingested from {self.raw_root} via IngestionProfile",
                image_count=len(archived),
                source=str(self.raw_root),
                tags=list(report.source_distribution.keys()),
            )
            snap.save(version_dir / "snapshot.yaml")

            # ── 6. Update registry ──
            asset = registry.get(dataset_name)
            if asset:
                asset.current_version = version_str
                asset.version_count = len(vm.list_versions()) + 1
                asset.total_images = len(archived)
                registry.update(asset)

            print(f"  [Auto-Register] Version {version_str} created ({len(archived)} images)")

            # ── 8. Write changelog entry ──
            try:
                from raven_trainer.data_asset.changelog import ChangelogStore
                ChangelogStore(R.dataset_dir(dataset_name)).log_version_created(
                    version_str,
                    detail=f"Auto-ingested {len(archived)} images from {self.target_dataset or dataset_name}",
                    count=len(archived),
                )
            except Exception as cl_err:
                print(f"  [Auto-Register] Changelog write error (non-fatal): {cl_err}")

            # ── 7. Write lineage ──
            try:
                ls = LineageStore(R.dataset_dir(dataset_name))
                # Add version node and edge from dataset
                version_id = f"{dataset_name}@{version_str}"
                ls.add_node(LineageNode(
                    id=version_id,
                    type="version",
                    name=version_str,
                    metadata={
                        "source": "ingestion",
                        "image_count": len(archived),
                        "raw_root": str(self.raw_root),
                    },
                ))
                ls.add_edge(LineageEdge(
                    source_id=dataset_name,
                    target_id=version_id,
                    relation="derived_from",
                ))
            except Exception as lineage_err:
                print(f"  [Auto-Register] Lineage write error (non-fatal): {lineage_err}")

        except Exception as e:
            print(f"  [Auto-Register Error] {e}")
            import traceback
            traceback.print_exc()
