"""Create Recipe for d6_person dataset and build training workspace.

This script:
1. Creates a Recipe that combines v1 (annotated) + v2 (video-extracted) 
2. Configures split strategy (by_video to prevent data leakage)
3. Builds the training workspace
4. Links the workspace to the project's dataset.yaml
"""
import yaml
import shutil
from pathlib import Path
from datetime import datetime

from raven_trainer.config import R

R.ensure_dirs()

DATASET_NAME = "d6_person"
PROJECT_NAME = "d6_person_intrusion"
RECIPE_NAME = "d6_train_v1v2"

# Paths
ds_path = R.dataset_dir(DATASET_NAME)
recipes_dir = ds_path / "recipes"
recipes_dir.mkdir(parents=True, exist_ok=True)
recipe_file = recipes_dir / f"recipe_{RECIPE_NAME}.yaml"

# Workspace output
workspace_dir = R.WORKSPACE / f"{DATASET_NAME}_{RECIPE_NAME}"

print("=" * 60)
print("D6 Person Dataset — Recipe & Workspace")
print("=" * 60)

# Step 1: Create Recipe YAML
print(f"\n[1/3] Creating Recipe: {RECIPE_NAME}")

recipe = {
    "name": RECIPE_NAME,
    "version": "1.0",
    "description": f"D6人员检测训练Recipe — 合并v1(标注)+v2(视频抽帧)数据",
    "created_at": datetime.now().isoformat(),
    
    # Data sources: combine v1 (annotated) and v2 (video-extracted)
    "sources": [
        {
            "name": f"{DATASET_NAME}@v1",
            "path": None,
            "include": [],
            "exclude": [],
        },
        {
            "name": f"{DATASET_NAME}@v2",
            "path": None,
            "include": [],
            "exclude": [],
        },
    ],
    
    "include": [],
    "exclude": [],
    
    # Sampling: use all data from v1, sample from v2
    "sampling": {
        "strategy": "all",
        "v1_weight": 1.0,
        "v2_weight": 0.5,
        "v2_as_negatives": True,  # v2 frames without annotations serve as hard negatives
    },
    
    # Split: by_video to prevent data leakage across cameras
    "split": {
        "strategy": "by_video",
        "train": 0.8,
        "val": 0.2,
        "test": 0.0,
        "seed": 42,
    },
    
    # Augmentation for detection
    "augment": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "flipud": 0.0,
        "fliplr": 0.5,
        "scale": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.3,
    },
    
    # Balance: limit per camera to prevent over-representation
    "balance": {
        "max_per_camera": 200,
        "time_balance": True,
    },
    
    # Quality filter
    "quality": {
        "min_blur_score": 50.0,
        "dedup_method": "dhash",
        "dedup_distance": 5,
    },
    
    "build_target": "yolo",
    "hard_negative_ratio": 0.05,
    
    "extra": {
        "project": PROJECT_NAME,
        "model": "yolo26s",
        "task": "detection",
        "classes": {"0": "person"},
    },
}

with open(recipe_file, "w", encoding="utf-8") as f:
    yaml.dump(recipe, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"  Recipe saved: {recipe_file}")
print(f"  Sources: v1 (annotated) + v2 (video-extracted)")
print(f"  Split: by_video (80/20, seed=42)")
print(f"  Augmentation: mosaic=1.0, mixup=0.1, copy_paste=0.3")

# Step 2: Build workspace from v1 annotated data
# Since v2 frames are unannotated, we build the workspace primarily from v1
# and add v2 frames as hard negatives
print(f"\n[2/3] Building Training Workspace")

if workspace_dir.exists():
    shutil.rmtree(workspace_dir)
workspace_dir.mkdir(parents=True, exist_ok=True)

# v1 data paths
v1_dir = ds_path / "versions" / "v1"
v1_images = v1_dir / "images"
v1_annotations = v1_dir / "annotations" / "converted" / "yolo"

# Check v1 structure
print(f"  v1 images dir: {v1_images.exists()}")
print(f"  v1 annotations dir: {v1_annotations.exists()}")

# If v1 images are stored as references (not copied), check manifest
manifest = v1_dir / "manifest.jsonl"
if manifest.exists():
    import json
    with open(manifest, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"  v1 manifest entries: {len(entries)}")
    
    # Check if images are in storage
    if entries:
        first = entries[0]
        print(f"  First entry keys: {list(first.keys())}")
        if "path" in first:
            print(f"  First image path: {first['path']}")

# Create workspace structure
ws_train = workspace_dir / "train"
ws_val = workspace_dir / "val"
ws_train_images = ws_train / "images"
ws_train_labels = ws_train / "labels"
ws_val_images = ws_val / "images"
ws_val_labels = ws_val / "labels"

for d in [ws_train_images, ws_train_labels, ws_val_images, ws_val_labels]:
    d.mkdir(parents=True, exist_ok=True)

print(f"  Workspace: {workspace_dir}")
print(f"  Structure: train/ val/ with images/ and labels/")

# Step 3: We'll use the dataset build command from CLI
# For now, just create the dataset.yaml that points to the workspace
print(f"\n[3/3] Creating workspace dataset.yaml")

workspace_yaml = {
    "path": str(workspace_dir).replace("\\", "/"),
    "train": "train/images",
    "val": "val/images",
    "test": "",
    "names": {"0": "person"},
    "task": "detection",
    "format": "yolo",
    "hard_negative_ratio": 0.05,
    "split_method": "video_based",
}

workspace_yaml_path = workspace_dir / "dataset.yaml"
with open(workspace_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(workspace_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"  dataset.yaml: {workspace_yaml_path}")
print(f"\n  Next step: Run 'raven-train dataset build d6_person -o {workspace_dir}' to populate")
print(f"  Or use the build_workspace.py script for full control")

print("\n" + "=" * 60)
print("Recipe & Workspace Setup Complete!")
print("=" * 60)
