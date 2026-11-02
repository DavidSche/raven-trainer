"""Import existing D6 YOLO training set as Asset Dataset v1.

Steps:
1. Copy all images (train+val) to data/storage/images/d6_original/
2. Copy all YOLO labels to version's annotations/converted/yolo/
3. Create manifest records for each image
4. Register as v1 using VersionManager
5. Update asset.yaml with current_version and total_images
"""
import shutil
import json
from pathlib import Path
from datetime import datetime

from raven_trainer.config import R
from raven_trainer.asset_manager import AssetManager
from raven_trainer.data_asset.registry import AssetStore, DatasetRegistry, DataAsset
from raven_trainer.data_asset.version import VersionManager, VersionSnapshot

# Ensure directories exist
R.ensure_dirs()

# Paths
old_dataset = Path(r"D:\D6-45\20260704_D6_0405-yolo-split")
storage_target = R.STORAGE_IMAGES / "d6_original"
storage_target.mkdir(parents=True, exist_ok=True)

dataset_name = "d6_person"
dataset_dir = R.dataset_dir(dataset_name)
version_manager = VersionManager(dataset_dir)

# Step 1: Copy images to storage
print("=== Step 1: Copying images to storage ===")
image_count = 0
label_count = 0
image_label_map = {}  # image_name -> label_path

for split in ["train", "val"]:
    img_dir = old_dataset / split / "images"
    label_dir = old_dataset / split / "labels"
    
    if not img_dir.exists():
        print(f"  [WARN] {split}/images not found, skipping")
        continue
    
    split_count = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        dest = storage_target / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
        image_label_map[img.name] = label_dir / (img.stem + ".txt")
        split_count += 1
    image_count += split_count
    print(f"  {split}: {split_count} images")

print(f"  Total images copied: {image_count}")

# Step 2: Create manifest records
print("\n=== Step 2: Creating manifest records ===")
am = AssetManager()
manifest_records = []

for img_name in sorted(image_label_map.keys()):
    # Storage path relative to project root
    storage_path = f"data/storage/images/d6_original/{img_name}"
    
    # Check if label exists
    label_path = image_label_map[img_name]
    has_annotation = label_path.exists() and label_path.stat().st_size > 0
    
    record = am.create_manifest_record(
        storage_path=storage_path,
        source="d6_original",
        image_id=f"d6_orig_{img_name[:30]}",
    )
    if has_annotation:
        record["has_annotation"] = True
        record["annotation_format"] = "yolo"
        label_count += 1
    else:
        record["has_annotation"] = False
    
    manifest_records.append(record)

print(f"  Manifest records: {len(manifest_records)}")
print(f"  With annotations: {label_count}")
print(f"  Without annotations: {len(manifest_records) - label_count}")

# Step 3: Register as v1
print("\n=== Step 3: Registering as v1 ===")
snapshot = version_manager.create_version(
    manifest_records=manifest_records,
    description="原始数据集导入（D6-45/20260704_D6_0405-yolo-split）",
    source="D6-45/20260704_D6_0405-yolo-split",
    version_tag="v1",
    tags=["original", "yolo-format"],
)
print(f"  Version created: {snapshot.version}")
print(f"  Image count: {snapshot.image_count}")

# Step 4: Copy YOLO labels to version's annotations directory
print("\n=== Step 4: Copying YOLO labels to version annotations ===")
version_dir = version_manager.get_version_dir("v1")
yolo_label_dir = version_dir / "annotations" / "converted" / "yolo"
yolo_label_dir.mkdir(parents=True, exist_ok=True)

copied_labels = 0
for img_name, label_path in sorted(image_label_map.items()):
    if label_path.exists() and label_path.stat().st_size > 0:
        dest_label = yolo_label_dir / (Path(img_name).stem + ".txt")
        if not dest_label.exists():
            shutil.copy2(label_path, dest_label)
        copied_labels += 1

print(f"  Labels copied: {copied_labels}")

# Also copy the dataset.yaml (class names) to the version
old_dataset_yaml = old_dataset / "dataset.yaml"
if old_dataset_yaml.exists():
    shutil.copy2(old_dataset_yaml, yolo_label_dir / "dataset.yaml")
    print("  Copied dataset.yaml (class definitions)")

# Step 5: Update asset.yaml
print("\n=== Step 5: Updating asset.yaml ===")
asset_store = AssetStore(R.DATASETS)
asset = asset_store.get_asset(dataset_name)
if asset:
    asset.current_version = "v1"
    asset.version_count = 1
    asset.total_images = image_count
    asset.total_annotations = label_count
    asset_store.update_asset(asset)
    print(f"  Updated: current_version=v1, total_images={image_count}, total_annotations={label_count}")
else:
    print("  [ERROR] Asset not found!")

# Step 6: Verify
print("\n=== Verification ===")
print(f"Storage images: {len(list(storage_target.glob('*')))}")
print(f"Version dir: {version_dir}")
print(f"  snapshot.yaml exists: {(version_dir / 'snapshot.yaml').exists()}")
print(f"  manifest.jsonl exists: {(version_dir / 'manifest.jsonl').exists()}")
print(f"  YOLO labels: {len(list(yolo_label_dir.glob('*.txt')))}")

# Read first few manifest records
manifest_path = version_dir / "manifest.jsonl"
if manifest_path.exists():
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"\nManifest first 3 records:")
    for line in lines[:3]:
        rec = json.loads(line)
        print(f"  {rec['image_id']}: {rec['storage_path']} (annotation: {rec.get('has_annotation', False)})")

# Read old dataset.yaml for class info
if old_dataset_yaml.exists():
    import yaml
    with open(old_dataset_yaml, 'r') as f:
        ds_config = yaml.safe_load(f)
    print(f"\nOriginal dataset classes: {ds_config.get('names', {})}")

print("\n=== Import Complete ===")
