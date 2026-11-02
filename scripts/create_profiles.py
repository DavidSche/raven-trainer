"""Create IngestionProfile YAML files for D6 video frame extraction."""
import yaml
from pathlib import Path
from raven_trainer.config import R
from raven_trainer.ingestion.profile import IngestionProfile, SourceRef, IngestionProfileManager

R.ensure_dirs()
mgr = IngestionProfileManager()

# Get all camera directories
video_base = R.STORAGE / "videos" / "d6_workshop"
cameras = sorted([d.name for d in video_base.iterdir() if d.is_dir()])
print(f"Found {len(cameras)} camera directories: {cameras}")

# ── Profile 1: d6_video_all — All cameras, comprehensive frame extraction ──
# Targets d6_person dataset, will be registered as v2
sources_all = [
    SourceRef(name=cam, path=f"videos/d6_workshop/{cam}", source_type="video")
    for cam in cameras
]

profile_all = IngestionProfile(
    name="d6_video_all",
    description="D6 all cameras - comprehensive frame extraction for person detection training",
    sources=sources_all,
    decode={
        "mode": "interval",
        "every_n_frames": 15,       # Dense sampling: every 15 frames (~0.5s at 30fps)
        "target_fps": 0.0,
        "jpeg_quality": 85,
    },
    pipeline=["motion", "scene_change", "burst", "balance"],
    sample_count=800,                # Target ~800 frames total
    sampling={
        "interval": {"interval": 15},
        "motion": {"motion_area_threshold": 0.3},  # Lower threshold to catch subtle movement
        "scene_change": {"method": "histogram", "threshold": 0.3},
        "burst": {"before": 3, "after": 5},
        "balance": {"max_per_camera": 50},  # Cap at 50 frames per camera for diversity
    },
    quality={
        "blur_laplacian_threshold": 100.0,
        "black_mean_threshold": 5.0,
        "overexposed_mean_threshold": 250.0,
        "overexposed_pixel_ratio": 0.5,
        "min_width": 320,
        "min_height": 240,
    },
    dedup={
        "method": "dhash",
        "hash_size": 8,
        "hash_distance": 5,
        "ssim_threshold": 0.98,
    },
    auto_register=True,
    target_dataset="d6_person",
)

path_all = mgr.save_profile(profile_all)
print(f"\nProfile 1 saved: {path_all}")
print(f"  Sources: {len(profile_all.sources)} cameras")
print(f"  Pipeline: {profile_all.pipeline}")
print(f"  Sample count: {profile_all.sample_count}")
print(f"  Target dataset: {profile_all.target_dataset}")

# ── Profile 2: d6_video_negatives — Background/negative samples ──
# Uses balance-only pipeline (no motion filtering) to get static/background frames
# Targets d6_negatives dataset
profile_neg = IngestionProfile(
    name="d6_video_negatives",
    description="D6 negative samples - background frames without motion filtering for FP reduction",
    sources=sources_all,
    decode={
        "mode": "interval",
        "every_n_frames": 60,       # Sparse sampling: every 60 frames (~2s at 30fps)
        "target_fps": 0.0,
        "jpeg_quality": 85,
    },
    pipeline=["balance"],            # Balance only - no motion/scene filtering
    sample_count=300,
    sampling={
        "interval": {"interval": 60},
        "balance": {"max_per_camera": 15},  # Cap at 15 frames per camera
    },
    quality={
        "blur_laplacian_threshold": 80.0,   # Slightly lower threshold for static scenes
        "black_mean_threshold": 5.0,
        "overexposed_mean_threshold": 250.0,
        "overexposed_pixel_ratio": 0.5,
        "min_width": 320,
        "min_height": 240,
    },
    dedup={
        "method": "dhash",
        "hash_size": 8,
        "hash_distance": 5,
        "ssim_threshold": 0.95,     # More aggressive dedup for static scenes
    },
    auto_register=True,
    target_dataset="d6_negatives",
)

path_neg = mgr.save_profile(profile_neg)
print(f"\nProfile 2 saved: {path_neg}")
print(f"  Sources: {len(profile_neg.sources)} cameras")
print(f"  Pipeline: {profile_neg.pipeline}")
print(f"  Sample count: {profile_neg.sample_count}")
print(f"  Target dataset: {profile_neg.target_dataset}")

# Verify
print("\n=== Saved profiles ===")
for name in mgr.list_profiles():
    p = mgr.get_profile(name)
    print(f"  {name}: {len(p.sources)} sources, pipeline={p.pipeline}, target={p.target_dataset}")
