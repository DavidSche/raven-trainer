"""Register extracted video frames as v2 of d6_person dataset.

This script:
1. Collects all extracted frames from profile_d6_video_all/{camera}/{video}/*.jpg
2. Copies them to a flat directory with unique names
3. Uses VersionManager to register as v2
4. Updates the dataset metadata
"""
import shutil
import time
from pathlib import Path
from datetime import datetime

from raven_trainer.config import R
from raven_trainer.data_asset.version import VersionManager

R.ensure_dirs()

# Paths
EXTRACTION_DIR = R.INGESTION_RUNS / "profile_d6_video_all"
FLAT_DIR = R.INGESTION_RUNS / "profile_d6_video_all_flat"
DATASET_NAME = "d6_person"

print("=" * 60)
print("D6 Person Dataset — v2 Registration")
print("=" * 60)

# Step 1: Collect all extracted frames into a flat directory
print(f"\n[1/3] Collecting frames from {EXTRACTION_DIR}")
if FLAT_DIR.exists():
    shutil.rmtree(FLAT_DIR)
FLAT_DIR.mkdir(parents=True)

collected = 0
cameras_with_frames = 0

for cam_dir in sorted(EXTRACTION_DIR.iterdir()):
    if not cam_dir.is_dir():
        continue
    
    cam_name = cam_dir.name
    cam_frames = list(cam_dir.rglob("*.jpg"))
    
    if cam_frames:
        cameras_with_frames += 1
        for frame in cam_frames:
            # Create unique name: camera_video_frame.jpg
            video_name = frame.parent.name
            frame_name = frame.stem  # e.g., 000000
            new_name = f"{cam_name}_{video_name}_{frame_name}.jpg"
            dest = FLAT_DIR / new_name
            shutil.copy2(frame, dest)
            collected += 1
    
    print(f"  {cam_name}: {len(cam_frames)} frames")

print(f"\nTotal collected: {collected} frames from {cameras_with_frames} cameras")
print(f"Flat directory: {FLAT_DIR}")

# Step 2: Register as v2 using VersionManager
print(f"\n[2/3] Registering as v2 of {DATASET_NAME}")
ds_path = R.dataset_dir(DATASET_NAME)
vm = VersionManager(ds_path)

# Check current versions
current_versions = vm.list_versions()
print(f"  Current versions: {current_versions}")

# Create new version
snapshot = vm.create_version(
    source_images=FLAT_DIR,
    description=f"视频抽帧数据（{collected}帧，来自{cameras_with_frames}个摄像头）",
    tags=["v2", "video_extraction", "unannotated"],
    source="video_frame_extraction",
)

print(f"  New version: {snapshot.version}")
print(f"  Image count: {snapshot.image_count}")
print(f"  Parent version: {snapshot.parent_version}")

# Step 3: Verify
print(f"\n[3/3] Verification")
versions = vm.list_versions()
print(f"  All versions: {versions}")
for v in versions:
    snap = vm.get_version(v)
    if snap:
        print(f"    {v}: {snap.image_count} images, {snap.annotation_count} annotations")

# Also update asset.yaml metadata
asset_yaml = ds_path / "asset.yaml"
if asset_yaml.exists():
    import yaml
    with open(asset_yaml, "r", encoding="utf-8") as f:
        asset = yaml.safe_load(f)
    
    if "versions" not in asset:
        asset["versions"] = []
    
    # Add v2 info
    v2_info = {
        "version": "v2",
        "description": f"视频抽帧数据（{collected}帧）",
        "image_count": collected,
        "source": "video_frame_extraction",
        "created_at": datetime.now().isoformat(),
        "tags": ["video_extraction", "unannotated"],
    }
    
    # Check if v2 already exists in the list
    existing_versions = [v.get("version") for v in asset.get("versions", [])]
    if "v2" not in existing_versions:
        asset["versions"].append(v2_info)
    
    asset["current_version"] = "v2"
    asset["total_images"] = sum(v.get("image_count", 0) for v in asset.get("versions", []))
    
    with open(asset_yaml, "w", encoding="utf-8") as f:
        yaml.dump(asset, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"\n  asset.yaml updated:")
    print(f"    current_version: {asset['current_version']}")
    print(f"    total_images: {asset['total_images']}")
    print(f"    versions: {len(asset['versions'])}")

print("\n" + "=" * 60)
print("v2 Registration Complete!")
print("=" * 60)
