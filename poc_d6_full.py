#!/usr/bin/env python3
"""d6-4-5 完整流程验证脚本。

运行方式: uv run python poc_d6_full.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

from raven_trainer.config import R
from raven_trainer.asset_manager import AssetManager
from raven_trainer.data_asset.registry import AssetStore, DatasetRegistry
from raven_trainer.data_asset.version import VersionSnapshot, VersionManager
from raven_trainer.data_asset.tag import TagManager
from raven_trainer.data_asset.lineage import LineageStore
from raven_trainer.data_asset.metadata import MetadataStore, ImageMetadata
from raven_trainer.data_asset.recipe import DatasetRecipe, RecipeManager
from raven_trainer.data_asset.intelligence import DatasetIntelligence
from raven_trainer.data_asset.doctor import DatasetDoctor
from raven_trainer.data_asset.annotation import AnnotationManager

NAME = "d6-4-5"
VERSION = "v1"
print(f"\n{'='*60}")
print(f"  Raven Trainer POC: {NAME}")
print(f"{'='*60}")

# ============================================================
# Step 1: Create dataset
# ============================================================
print(f"\n[Step 1/12] Creating dataset...")
store = AssetStore(R.DATASETS)
registry = DatasetRegistry(store)
try:
    registry.create(NAME, task="detection", display_name="D6-4-5 People", owner="POC")
    print("  Created")
except FileExistsError:
    print("  Already exists")

# ============================================================
# Step 2: Register Manifest
# ============================================================
print(f"\n[Step 2/12] Registering Manifest...")
am = AssetManager()
records = []
project_root = Path(R.PROJECT_ROOT).resolve()
src_dir = R.STORAGE_IMAGES / NAME
files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in (".jpg",".jpeg",".png")])
print(f"  Scanning {len(files)} images...")

for i, f in enumerate(files):
    rel = f.resolve().relative_to(project_root)
    record = am.create_manifest_record(storage_path=str(rel.as_posix()), source=NAME, image_id=f"d6_{i:04d}")
    records.append(record)

manifest_path = R.dataset_manifest_path(NAME, VERSION)
am.save_manifest(records, manifest_path)
print(f"  Manifest: {len(records)} records")

# Version + Registry
snap = VersionSnapshot(version=VERSION, image_count=len(records), source=f"storage/images/{NAME}")
snap.save(R.dataset_version_dir(NAME, VERSION) / "snapshot.yaml")
asset = registry.get(NAME)
asset.current_version = VERSION
asset.version_count = 1
asset.total_images = len(records)
registry.update(asset)
print(f"  Registry: v1 ({asset.total_images} images)")

# ============================================================
# Step 3: Import YOLO labels
# ============================================================
print(f"\n[Step 3/12] Importing YOLO labels...")
src_train = Path(r"E:\work-space\20260228\D6-4_5_people-refresh-20260625-yolo-split\train\labels")
src_val = Path(r"E:\work-space\20260228\D6-4_5_people-refresh-20260625-yolo-split\val\labels")
ann_dir = R.dataset_version_dir(NAME, VERSION) / "annotations" / "converted" / "yolo"
ann_dir.mkdir(parents=True, exist_ok=True)

# Build filename -> manifest index mapping
name_map = {}
with open(manifest_path) as mf:
    manifest_recs = [json.loads(l) for l in mf if l.strip()]
for i, rec in enumerate(manifest_recs):
    stem = Path(rec["storage_path"]).stem
    name_map[stem] = f"{i:06d}"

copied = 0
for src in [src_train, src_val]:
    for f in src.glob("*.txt"):
        mapped = name_map.get(f.stem)
        if mapped:
            shutil.copy2(str(f), str(ann_dir / f"{mapped}.txt"))
            copied += 1
print(f"  Copied {copied} label files")
asset.total_annotations = copied
registry.update(asset)

# Save name mapping
with open(ann_dir.parent / "name_map.json", "w") as f:
    json.dump(name_map, f, indent=2)

# ============================================================
# Step 4: tags.yaml
# ============================================================
print(f"\n[Step 4/12] Creating tags.yaml...")
tm = TagManager(R.dataset_dir(NAME))
tm.define("people", category="class", description="People detection")
tm.define("single_class", category="task", description="Single class dataset")
print("  OK")

# ============================================================
# Step 5: lineage.yaml
# ============================================================
print(f"\n[Step 5/12] Creating lineage.yaml...")
ls = LineageStore(R.dataset_dir(NAME))
ls.register_chain(
    video_id=f"storage_{NAME}",
    image_id=f"manifest_{NAME}_{VERSION}",
    dataset_name=f"dataset_{NAME}",
    version=f"{NAME}_{VERSION}",
    recipe="recipe_default",
)
print("  OK")

# ============================================================
# Step 6: metadata (image, quality, source)
# ============================================================
print(f"\n[Step 6/12] Creating metadata...")
ms = MetadataStore(R.dataset_dir(NAME) / "metadata")
with open(manifest_path) as mf:
    manifest_recs = [json.loads(l) for l in mf if l.strip()]
for i, rec in enumerate(manifest_recs):
    meta = ImageMetadata(
        image_id=rec.get("image_id", f"d6_{i:04d}"),
        filename=Path(rec.get("storage_path","")).name,
        camera=rec.get("source", NAME),
        frame_index=i,
    )
    label_file = ann_dir / f"{i:06d}.txt"
    if label_file.exists() and label_file.stat().st_size > 0:
        with open(label_file) as f:
            meta.label_count = len([l for l in f if l.strip()])
        meta.has_annotation = meta.label_count > 0
    ms.add(meta)
print(f"  {ms.count()} records written")

# ============================================================
# Step 7: annotations/raw/
# ============================================================
print(f"\n[Step 7/12] Creating annotations/raw/...")
raw_dir = R.dataset_version_dir(NAME, VERSION) / "annotations" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
(raw_dir / ".gitkeep").write_text("")
print("  OK")

# ============================================================
# Step 8: recipe
# ============================================================
print(f"\n[Step 8/12] Creating recipe...")
rm = RecipeManager(R.dataset_dir(NAME))
rm.save_recipe(DatasetRecipe(name=NAME, description="Single class people detection", build_target="yolo"))
print("  OK")

# ============================================================
# Step 9: Dataset Intelligence
# ============================================================
print(f"\n[Step 9/12] Dataset Intelligence...")
di = DatasetIntelligence(NAME, VERSION)
report = di.analyze()
di.save_report(report)
print(report.summary())

# Generate HTML report
def pct(v, total):
    return f"{v/max(total,1)*100:.1f}%"

max_sz_val = max(report.small_object_ratio, report.medium_object_ratio, report.large_object_ratio) or 1

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Dataset Intelligence - {NAME}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:-apple-system,sans-serif;background:#f5f6fa;color:#2c3e50;padding:40px;}}
.container{{max-width:960px;margin:0 auto;}}
h1{{font-size:24px;}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin:20px 0;}}
.stat-card{{background:white;border-radius:8px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
.stat-card .value{{font-size:28px;font-weight:700;}}
.stat-card .label{{font-size:13px;color:#666;}}
.card{{background:white;border-radius:8px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
.card h2{{font-size:16px;margin-bottom:12px;border-bottom:1px solid #eee;padding-bottom:8px;}}
table{{width:100%;border-collapse:collapse;}}
th{{text-align:left;font-size:12px;color:#666;padding:6px 4px;border-bottom:2px solid #eee;}}
td{{padding:6px 4px;border-bottom:1px solid #f0f0f0;font-size:14px;}}
.bar{{height:8px;background:#4A90D9;border-radius:4px;}}
.warn{{color:#E67E22;}}
</style></head>
<body><div class="container">
<h1>Dataset Intelligence Report</h1>
<p style="color:#666;margin-bottom:20px">{NAME} v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="stats-grid">
<div class="stat-card"><div class="value">{report.total_images:,}</div><div class="label">Total Images</div></div>
<div class="stat-card"><div class="value">{report.total_annotations:,}</div><div class="label">Annotations</div></div>
<div class="stat-card"><div class="value">{pct(report.annotated_images, report.total_images)}</div><div class="label">Annotated</div></div>
<div class="stat-card"><div class="value">{pct(report.background_images, report.total_images)}</div><div class="label">Background</div></div>
<div class="stat-card"><div class="value">{report.small_object_ratio:.1f}%</div><div class="label">Small Objects</div></div>
<div class="stat-card"><div class="value">{report.medium_object_ratio:.1f}%</div><div class="label">Medium Objects</div></div>
</div>"""

# Size
html += '<div class="card"><h2>Object Size</h2><table><tr><th>Size</th><th>Ratio</th><th></th></tr>'
for label, ratio in [("Small (<1%)", report.small_object_ratio), ("Medium (1-10%)", report.medium_object_ratio), ("Large (>10%)", report.large_object_ratio)]:
    html += f'<tr><td>{label}</td><td>{ratio:.1f}%</td><td><div class="bar" style="width:{ratio/max_sz_val*100:.1f}%"></div></td></tr>'
html += "</table></div>"

# Camera
if report.camera_distribution:
    html += '<div class="card"><h2>Camera Distribution</h2><table><tr><th>Camera</th><th>Count</th><th>Ratio</th><th></th></tr>'
    max_cam = max(report.camera_distribution.values())
    for cam, count in sorted(report.camera_distribution.items(), key=lambda x: -x[1]):
        html += f'<tr><td>{cam}</td><td>{count}</td><td>{pct(count, report.total_images)}</td><td><div class="bar" style="width:{count/max_cam*100:.1f}%"></div></td></tr>'
    html += "</table></div>"

# Class
if report.class_distribution:
    html += '<div class="card"><h2>Class Distribution</h2><table><tr><th>Class</th><th>Count</th><th>Ratio</th><th></th></tr>'
    max_cls = max(report.class_distribution.values())
    for cls_id, count in sorted(report.class_distribution.items(), key=lambda x: -x[1])[:15]:
        html += f'<tr><td>{cls_id}</td><td>{count}</td><td>{pct(count, report.total_annotations)}</td><td><div class="bar" style="width:{count/max_cls*100:.1f}%"></div></td></tr>'
    html += "</table></div>"

# Biases
if report.bias_warnings:
    html += '<div class="card"><h2>Bias Warnings</h2>'
    for w in report.bias_warnings:
        html += f'<div class="warn">! {w}</div>'
    html += "</div>"
if report.recommendations:
    html += '<div class="card"><h2>Recommendations</h2><ul>'
    for r in report.recommendations:
        html += f"<li>{r}</li>"
    html += "</ul></div>"

html += "</div></body></html>"

reports_dir = R.dataset_dir(NAME) / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
html_path = reports_dir / "intelligence_report.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML Report: {html_path} ({html_path.stat().st_size}B)")

# ============================================================
# Step 10: Dataset Doctor
# ============================================================
print(f"\n[Step 10/12] Dataset Doctor...")
vm = VersionManager(R.dataset_dir(NAME))
versions = vm.list_versions()
manifest_path = R.dataset_manifest_path(NAME, versions[0].version) if versions else None
ann_dir2 = R.dataset_version_dir(NAME, versions[0].version) / "annotations" / "converted" / "yolo" if versions else None

doctor = DatasetDoctor(R.dataset_dir(NAME))
doctor_report = doctor.check(
    metadata_count=report.total_images,
    source_distribution=report.camera_distribution,
    manifest_path=manifest_path,
    ann_dir=ann_dir2 if ann_dir2 and ann_dir2.exists() else None,
)
doctor.save_report(doctor_report)
print(f"  {doctor_report.summary}")

# ============================================================
# Step 11: Create training project
# ============================================================
print(f"\n[Step 11/12] Creating training project...")
# Write project files directly
proj_dir = R.PROJECT_ROOT / "projects" / "d6_people"
proj_dir.mkdir(parents=True, exist_ok=True)

import yaml
# project.yaml
yaml.dump({
    "name": "d6_people", "version": "1.0", "task": "detection",
    "backend": {"name": "ultralytics", "model": "yolo26s"},
    "dataset": {"source": NAME, "version": VERSION, "ref": str(R.dataset_dir(NAME))},
    "export": {"format": "onnx", "opset": 17},
}, open(proj_dir / "project.yaml", "w"), default_flow_style=False)

# dataset.yaml
yaml.dump({
    "path": str(Path(r"E:\work-space\20260228\D6-4_5_people-refresh-20260625-yolo-split")),
    "train": "train/images", "val": "val/images",
    "nc": 1, "names": {"0": "people"},
}, open(proj_dir / "dataset.yaml", "w"), default_flow_style=False)

# train.yaml
yaml.dump({
    "epochs": 50, "batch": 16, "imgsz": 640, "optimizer": "AdamW",
    "lr": 0.001, "lr_scheduler": "cosine", "amp": True, "patience": 20,
}, open(proj_dir / "train.yaml", "w"), default_flow_style=False)

# Create subdirs
for d in ["experiments", "models", "outputs"]:
    (proj_dir / d).mkdir(exist_ok=True)

print(f"  Project: {proj_dir}")

# ============================================================
# Step 12: Build workspace
# ============================================================
print(f"\n[Step 12/12] Building workspace...")
ws_dir = R.workspace_dir(f"{NAME}_built")
ws_dir.mkdir(parents=True, exist_ok=True)
yaml.dump({
    "path": str(Path(r"E:\work-space\20260228\D6-4_5_people-refresh-20260625-yolo-split")),
    "train": "train/images", "val": "val/images",
    "nc": 1, "names": {"0": "people"},
}, open(ws_dir / "dataset.yaml", "w"), default_flow_style=False)
print(f"  Workspace: {ws_dir}")

# ============================================================
# Final verification
# ============================================================
print(f"\n{'='*60}")
print(f"  Final Verification")
print(f"{'='*60}")

checks = [
    ("asset.yaml", R.dataset_dir(NAME) / "asset.yaml"),
    ("versions/v1/snapshot.yaml", R.dataset_version_dir(NAME, VERSION) / "snapshot.yaml"),
    ("versions/v1/manifest.jsonl", R.dataset_manifest_path(NAME, VERSION)),
    ("annotations/converted/yolo/", R.dataset_version_dir(NAME, VERSION) / "annotations" / "converted" / "yolo"),
    ("annotations/raw/", R.dataset_version_dir(NAME, VERSION) / "annotations" / "raw"),
    ("tags.yaml", R.dataset_dir(NAME) / "tags.yaml"),
    ("lineage.yaml", R.dataset_dir(NAME) / "lineage.yaml"),
    ("metadata/image.jsonl", R.dataset_dir(NAME) / "metadata" / "image.jsonl"),
    ("metadata/quality.jsonl", R.dataset_dir(NAME) / "metadata" / "quality.jsonl"),
    ("metadata/source.jsonl", R.dataset_dir(NAME) / "metadata" / "source.jsonl"),
    ("recipes/recipe_default.yaml", R.dataset_dir(NAME) / "recipes" / "recipe_default.yaml"),
    ("reports/intelligence_report.html", R.dataset_dir(NAME) / "reports" / "intelligence_report.html"),
    ("reports/health_latest.json", R.dataset_dir(NAME) / "reports" / "health_latest.json"),
    ("projects/d6_people/project.yaml", R.PROJECT_ROOT / "projects" / "d6_people" / "project.yaml"),
    ("projects/d6_people/dataset.yaml", R.PROJECT_ROOT / "projects" / "d6_people" / "dataset.yaml"),
    ("projects/d6_people/train.yaml", R.PROJECT_ROOT / "projects" / "d6_people" / "train.yaml"),
    ("workspace/dataset.yaml", R.workspace_dir(f"{NAME}_built") / "dataset.yaml"),
]

all_pass = True
for label, path in checks:
    exists = path.exists() if hasattr(path, "exists") else False
    status = "PASS" if exists else "FAIL"
    if not exists:
        all_pass = False
    print(f"  [{status}] {label}")

print(f"\n  Dataset: {NAME} v{VERSION}")
print(f"  Images: {asset.total_images}, Labels: {asset.total_annotations}")
print(f"  Intelligence OK: {report.total_images} images, {report.total_annotations} annotations")
print(f"  Doctor Score: {doctor_report.overall_score}/100")
print()
print(f"  Training command:")
print(f"    raven-train train projects/d6_people/project.yaml --epochs 50")
print()
if all_pass:
    print(f"  {'='*30}")
    print(f"   ALL CHECKS PASSED")
    print(f"  {'='*30}")
else:
    print(f"  SOME CHECKS FAILED")
