"""Run video frame extraction for all D6 cameras.

The RawImporter only scans immediate children of raw_root, so we need to
run the pipeline per-camera-directory. This script:
1. Iterates over all camera directories under data/storage/videos/d6_workshop/
2. For each camera, creates and runs an IngestionPipeline
3. Collects all extracted frames into a single output directory
4. Registers the results as a new version (v2) of d6_person dataset
"""
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

from raven_trainer.config import R
from raven_trainer.ingestion.core import QualityFilterConfig, DedupConfig
from raven_trainer.ingestion.pipeline import IngestionPipeline
from raven_trainer.ingestion.sampling.core import DecodeConfig
from raven_trainer.ingestion.importers import RawImporter

R.ensure_dirs()

# Configuration
VIDEO_BASE = R.STORAGE / "videos" / "d6_workshop"
OUTPUT_DIR = R.INGESTION_RUNS / "profile_d6_video_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline parameters (from d6_video_all profile)
PIPELINE_STAGES = ["motion", "scene_change", "burst", "balance"]
SAMPLE_COUNT = 50          # per camera (50 cameras × ~27 = ~1350 max)
BLUR_THRESHOLD = 100.0
DEDUP_METHOD = "dhash"
DEDUP_DISTANCE = 5
EVERY_N_FRAMES = 15
JPEG_QUALITY = 85

# Sampling config
SAMPLER_CONFIG = {
    "interval": {"interval": EVERY_N_FRAMES},
    "motion": {"motion_area_threshold": 0.3},
    "scene_change": {"method": "histogram", "threshold": 0.3},
    "burst": {"before": 3, "after": 5},
    "balance": {"max_per_camera": SAMPLE_COUNT},
}

QUALITY = QualityFilterConfig(
    blur_laplacian_threshold=BLUR_THRESHOLD,
    black_mean_threshold=5.0,
    overexposed_mean_threshold=250.0,
    overexposed_pixel_ratio=0.5,
    min_width=320,
    min_height=240,
)

DEDUP = DedupConfig(
    method=DEDUP_METHOD,
    hash_distance=DEDUP_DISTANCE,
)

DECODE = DecodeConfig(
    mode="interval",
    every_n_frames=EVERY_N_FRAMES,
    target_fps=0.0,
    jpeg_quality=JPEG_QUALITY,
)

# Get all camera directories
cameras = sorted([d for d in VIDEO_BASE.iterdir() if d.is_dir() and any(d.glob("*.mp4"))])
print(f"Found {len(cameras)} camera directories with videos")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Pipeline: {' -> '.join(PIPELINE_STAGES)}")
print(f"Sample count per camera: {SAMPLE_COUNT}")
print(f"Every N frames: {EVERY_N_FRAMES}")
print()

# Process each camera
all_results = []
total_extracted = 0
total_quality_removed = 0
total_dedup_removed = 0
failed_cameras = []
t_start = time.time()

for i, cam_dir in enumerate(cameras, 1):
    cam_name = cam_dir.name
    print(f"\n[{i}/{len(cameras)}] Processing camera: {cam_name}")
    
    # Count videos in this camera
    videos = sorted(cam_dir.glob("*.mp4"))
    print(f"  Videos: {len(videos)}")
    
    try:
        # Create and run pipeline for this camera
        pipeline = IngestionPipeline(
            raw_root=cam_dir,
            sampling_pipeline=PIPELINE_STAGES,
            sampler_config=SAMPLER_CONFIG,
            sample_count=SAMPLE_COUNT,
            quality=QUALITY,
            dedup=DEDUP,
            decode_config=DECODE,
            auto_register=False,  # We'll register manually after all cameras
            target_dataset="d6_person",
        )
        
        # Run with output to a camera-specific subdirectory
        cam_output = OUTPUT_DIR / cam_name
        report = pipeline.run(
            output_dir=cam_output,
            skip_scene=True,
            generate_report=False,  # Skip per-camera reports
        )
        
        kept = report.total_after_dedup
        total_extracted += kept
        total_quality_removed += (
            report.quality_removed_blurry + report.quality_removed_black +
            report.quality_removed_overexposed + report.quality_removed_small
        )
        total_dedup_removed += report.dedup_removed
        
        all_results.append({
            "camera": cam_name,
            "videos": len(videos),
            "kept": kept,
            "quality_removed": report.quality_removed_blurry + report.quality_removed_black + 
                               report.quality_removed_overexposed + report.quality_removed_small,
            "dedup_removed": report.dedup_removed,
        })
        
        print(f"  Kept: {kept} frames")
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed_cameras.append(cam_name)
        all_results.append({
            "camera": cam_name,
            "videos": len(videos),
            "kept": 0,
            "error": str(e),
        })

elapsed = time.time() - t_start

# Summary
print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)
print(f"Total cameras processed: {len(cameras)}")
print(f"Failed cameras: {len(failed_cameras)}")
if failed_cameras:
    print(f"  Failed: {failed_cameras}")
print(f"Total frames extracted: {total_extracted}")
print(f"Total quality-removed: {total_quality_removed}")
print(f"Total dedup-removed: {total_dedup_removed}")
print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

# Per-camera distribution
print("\nPer-camera distribution:")
for r in sorted(all_results, key=lambda x: -x.get("kept", 0)):
    status = "OK" if r.get("kept", 0) > 0 else "FAIL"
    print(f"  {r['camera']:15s}  videos={r['videos']:3d}  kept={r.get('kept', 0):4d}  [{status}]")

# Save results JSON
results_json = {
    "timestamp": datetime.now().isoformat(),
    "total_cameras": len(cameras),
    "total_extracted": total_extracted,
    "total_quality_removed": total_quality_removed,
    "total_dedup_removed": total_dedup_removed,
    "elapsed_sec": round(elapsed, 1),
    "failed_cameras": failed_cameras,
    "per_camera": all_results,
}
results_path = OUTPUT_DIR / "extraction_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {results_path}")

# Count total output images
total_images = 0
for cam_dir in OUTPUT_DIR.iterdir():
    if cam_dir.is_dir():
        total_images += len(list(cam_dir.glob("*.jpg")))
print(f"Total output images: {total_images}")
