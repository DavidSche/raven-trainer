"""Build training workspace from v1 annotated data + v2 extracted frames.

This script creates a YOLO-format training workspace:
1. Reads v1 manifest to get all annotated images
2. Groups images by camera (extracted from filename)
3. Splits by camera group (80/20) to prevent data leakage
4. Copies images and labels to workspace train/ and val/
5. Adds v2 frames as hard negatives (empty label files)
6. Creates dataset.yaml for training
"""
import json
import re
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

from raven_trainer.config import R

R.ensure_dirs()

# Configuration
DATASET_NAME = "d6_person"
WORKSPACE_DIR = R.WORKSPACE / "d6_train_v1v2"
VAL_RATIO = 0.2
SEED = 42
HARD_NEG_RATIO = 0.05  # 5% hard negatives from v2

random.seed(SEED)

print("=" * 60)
print("D6 Person — Build Training Workspace")
print("=" * 60)

# Paths
base = R.ROOT
v1_dir = R.dataset_dir(DATASET_NAME) / "versions" / "v1"
manifest_path = v1_dir / "manifest.jsonl"
annotations_dir = v1_dir / "annotations" / "converted" / "yolo"
storage_images = R.STORAGE / "images" / "d6_original"

# v2 extraction output
v2_dir = R.INGESTION_RUNS / "profile_d6_video_all"

# Step 1: Read manifest and group by camera
print(f"\n[1/5] Reading v1 manifest...")
with open(manifest_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f if line.strip()]

print(f"  Total entries: {len(entries)}")
annotated = [e for e in entries if e.get("has_annotation")]
unannotated = [e for e in entries if not e.get("has_annotation")]
print(f"  Annotated: {len(annotated)}")
print(f"  Unannotated (negatives): {len(unannotated)}")

# Extract camera ID from filename
# Pattern: 04-H01-01-000001.jpeg -> camera = 04-H01-01
# Pattern: 05-H06-03_20260209_10.jpg -> camera = 05-H06-03
def extract_camera(filename):
    # Match patterns like XX-HXX-XX or XX-XX-XX
    m = re.match(r'(\d{2}-[A-Z]\d{2}-\d{2})', filename)
    if m:
        return m.group(1)
    # Match patterns like XX-XX-XX (numeric)
    m = re.match(r'(\d{2}-\d{2}-\d{2})', filename)
    if m:
        return m.group(1)
    # Match other patterns
    m = re.match(r'(\d{2}-\w+)', filename)
    if m:
        return m.group(1)
    return "unknown"

camera_groups = defaultdict(list)
for entry in annotated:
    img_name = Path(entry["storage_path"]).name
    camera = extract_camera(img_name)
    camera_groups[camera].append(entry)

print(f"\n  Camera groups: {len(camera_groups)}")
for cam, items in sorted(camera_groups.items()):
    print(f"    {cam}: {len(items)} images")

# Step 2: Split by camera (group-based split to prevent leakage)
print(f"\n[2/5] Splitting by camera (val_ratio={VAL_RATIO})...")
cameras = sorted(camera_groups.keys())
random.shuffle(cameras)

val_camera_count = max(1, int(len(cameras) * VAL_RATIO))
val_cameras = set(cameras[:val_camera_count])
train_cameras = set(cameras[val_camera_count:])

print(f"  Train cameras ({len(train_cameras)}): {sorted(train_cameras)}")
print(f"  Val cameras ({len(val_cameras)}): {sorted(val_cameras)}")

train_entries = []
val_entries = []
for cam in train_cameras:
    train_entries.extend(camera_groups[cam])
for cam in val_cameras:
    val_entries.extend(camera_groups[cam])

# Also add some unannotated v1 images as negatives to train
neg_count = min(len(unannotated), int(len(train_entries) * HARD_NEG_RATIO))
train_negatives = random.sample(unannotated, neg_count) if neg_count > 0 else []

print(f"  Train: {len(train_entries)} annotated + {len(train_negatives)} negatives = {len(train_entries) + len(train_negatives)}")
print(f"  Val: {len(val_entries)} annotated")

# Step 3: Create workspace structure
print(f"\n[3/5] Creating workspace at {WORKSPACE_DIR}...")
if WORKSPACE_DIR.exists():
    shutil.rmtree(WORKSPACE_DIR)

# YOLO directory structure
ws_train_img = WORKSPACE_DIR / "train" / "images"
ws_train_lbl = WORKSPACE_DIR / "train" / "labels"
ws_val_img = WORKSPACE_DIR / "val" / "images"
ws_val_lbl = WORKSPACE_DIR / "val" / "labels"

for d in [ws_train_img, ws_train_lbl, ws_val_img, ws_val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Step 4: Copy images and labels
print(f"\n[4/5] Copying images and labels...")

def copy_entry(entry, img_dest_dir, lbl_dest_dir, is_negative=False):
    """Copy an image and its label to the workspace."""
    img_name = Path(entry["storage_path"]).name
    img_src = base / entry["storage_path"]
    
    if not img_src.exists():
        return False
    
    # Copy image
    img_dst = img_dest_dir / img_name
    if not img_dst.exists():
        shutil.copy2(img_src, img_dst)
    
    # Copy or create label
    lbl_name = Path(img_name).stem + ".txt"
    lbl_dst = lbl_dest_dir / lbl_name
    
    if is_negative:
        # Create empty label file for negatives
        lbl_dst.write_text("")
    else:
        # Find corresponding label in annotations
        lbl_src = annotations_dir / lbl_name
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            # Try with .txt extension variations
            lbl_src = annotations_dir / (Path(img_name).stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                # No label found, create empty
                lbl_dst.write_text("")
    
    return True

# Copy train data
train_copied = 0
for entry in train_entries:
    if copy_entry(entry, ws_train_img, ws_train_lbl):
        train_copied += 1

# Copy train negatives
train_neg_copied = 0
for entry in train_negatives:
    if copy_entry(entry, ws_train_img, ws_train_lbl, is_negative=True):
        train_neg_copied += 1

# Copy val data
val_copied = 0
for entry in val_entries:
    if copy_entry(entry, ws_val_img, ws_val_lbl):
        val_copied += 1

print(f"  Train: {train_copied} annotated + {train_neg_copied} negatives = {train_copied + train_neg_copied}")
print(f"  Val: {val_copied} annotated")

# Step 5: Add v2 frames as hard negatives (if extraction is done)
print(f"\n[5/5] Adding v2 frames as hard negatives...")
v2_added = 0
if v2_dir.exists():
    v2_frames = list(v2_dir.rglob("*.jpg"))
    print(f"  v2 frames available: {len(v2_frames)}")
    
    if v2_frames:
        # Sample a portion as hard negatives
        v2_neg_count = min(len(v2_frames), int(train_copied * HARD_NEG_RATIO))
        v2_sample = random.sample(v2_frames, v2_neg_count) if v2_neg_count > 0 else []
        
        for i, frame in enumerate(v2_sample):
            # Create unique name
            new_name = f"v2_{i:06d}.jpg"
            img_dst = ws_train_img / new_name
            lbl_dst = ws_train_lbl / f"v2_{i:06d}.txt"
            
            shutil.copy2(frame, img_dst)
            lbl_dst.write_text("")  # Empty label = negative sample
            v2_added += 1
        
        print(f"  v2 hard negatives added: {v2_added}")
    else:
        print(f"  No v2 frames found (extraction may still be running)")
else:
    print(f"  v2 directory not found")

# Create dataset.yaml
dataset_yaml = {
    "path": str(WORKSPACE_DIR).replace("\\", "/"),
    "train": "train/images",
    "val": "val/images",
    "test": "",
    "names": {0: "person"},
    "nc": 1,
    "task": "detection",
    "format": "yolo",
    "hard_negative_ratio": HARD_NEG_RATIO,
    "split_method": "video_based",
}

yaml_path = WORKSPACE_DIR / "dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

# Summary
print("\n" + "=" * 60)
print("Workspace Build Complete!")
print("=" * 60)
print(f"\nWorkspace: {WORKSPACE_DIR}")
print(f"  train/images: {len(list(ws_train_img.iterdir()))} files")
print(f"  train/labels: {len(list(ws_train_lbl.iterdir()))} files")
print(f"  val/images:   {len(list(ws_val_img.iterdir()))} files")
print(f"  val/labels:   {len(list(ws_val_lbl.iterdir()))} files")
print(f"  dataset.yaml: {yaml_path}")

# Verify label coverage
train_imgs = set(f.stem for f in ws_train_img.iterdir())
train_lbls = set(f.stem for f in ws_train_lbl.iterdir())
missing_lbls = train_imgs - train_lbls
extra_lbls = train_lbls - train_imgs
print(f"\n  Label coverage:")
print(f"    Images without labels: {len(missing_lbls)} (negatives)")
print(f"    Labels without images: {len(extra_lbls)}")

# Count non-empty labels
non_empty = sum(1 for f in ws_train_lbl.iterdir() if f.stat().st_size > 0)
print(f"    Non-empty labels: {non_empty}")
print(f"    Empty labels (negatives): {len(train_lbls) - non_empty}")
