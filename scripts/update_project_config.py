"""Update project dataset.yaml to point to the new workspace."""
from pathlib import Path
import yaml

base = Path(r"E:\study-place\Davidche\2026\raven\raven-trainer")
workspace = base / "data/workspace/d6_train_v1v2"
project_ds = base / "projects/d6_person_intrusion/dataset.yaml"

# Read workspace dataset.yaml
with open(workspace / "dataset.yaml", "r", encoding="utf-8") as f:
    ws_config = yaml.safe_load(f)

# Update project dataset.yaml
project_config = {
    "path": str(workspace).replace("\\", "/"),
    "train": "train/images",
    "val": "val/images",
    "test": "",
    "names": {0: "person"},
    "nc": 1,
    "task": "detection",
    "format": "yolo",
    "hard_negative_ratio": 0.05,
    "split_method": "video_based",
}

with open(project_ds, "w", encoding="utf-8") as f:
    yaml.dump(project_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"Updated: {project_ds}")
print(f"  path: {project_config['path']}")
print(f"  train: {project_config['train']}")
print(f"  val: {project_config['val']}")
print(f"  names: {project_config['names']}")

# Also show the current train.yaml
train_yaml = base / "projects/d6_person_intrusion/train.yaml"
print(f"\nCurrent train.yaml:")
with open(train_yaml, "r", encoding="utf-8") as f:
    print(f.read())

# Show project.yaml
project_yaml = base / "projects/d6_person_intrusion/project.yaml"
print(f"\nCurrent project.yaml:")
with open(project_yaml, "r", encoding="utf-8") as f:
    print(f.read())
