# Raven Trainer

**Raven AI Training Platform** — Industrial Vision Dataset & Model Training Platform.

Raven Trainer is the training counterpart of Raven-Go. It is a **platform** focused on
**dataset pipeline** (≈60% of the code) and model training/export.

## Architecture

```
              Raven AI Platform

+--------------------------------------------------+
|                   Raven-Go                        |
|        Runtime / Flow / Engine                    |
+--------------------------------------------------+
                         ↑ ONNX + Metadata
+--------------------------------------------------+
|            raven-onnxruntime                      |
|    Inference / Eval / Benchmark                   |
+--------------------------------------------------+
                         ↑ ONNX + Metadata
+--------------------------------------------------+
|              Raven-Trainer                        |
|  Dataset Pipeline (60%) / Trainer / Exporter     |
+--------------------------------------------------+
                         ↑
+--------------------------------------------------+
|    person_intrusion / helmet / ppe / smoke       |
|       (projects: only configs, no code)          |
+--------------------------------------------------+
```

## Dataset Pipeline (7 Layers)

```
Raw Video
    ↓
Import          Frame extraction, media import
    ↓
Cleaning        Dedup (hash/SSIM), blurry/black/overexposed removal
    ↓
Annotation      Storage in CVAT XML / COCO JSON (NOT YOLO)
    ↓
Version         Versioned snapshots with manifest
    ↓
Dataset Builder Plugin system: YOLO / COCO / Pose / Seg
    ↓
Workspace       train/val split, ready for trainer
    ↓
Trainer         Model training via pluggable adapters
    ↓
Exporter        ONNX + Metadata
```

The key insight: **Dataset Recipe** is the "source code" of a training dataset.
Maintain the recipe, not the data. The same recipe always produces the same dataset.

## CLI Usage

```bash
# Create a project (model: yolo26n/s/m/l/x, yolo11n/s/m/l/x, rtdetr-l;
#   rf-detr-nano/small/medium/large + seg-* with --backend rfdetr)
raven-train new --name person_intrusion --task detection --model yolo26n

# Dataset pipeline
raven-train dataset import video.mp4 --output ./datasets/imported
raven-train dataset clean ./datasets/imported --output ./datasets/cleaned
raven-train dataset version person_v1 --create
raven-train dataset build dataset_recipe.yaml --format yolo

# Train & Export
raven-train train projects/person_intrusion/project.yaml
raven-train export projects/person_intrusion/project.yaml

# OBB (rotated boxes): always use the wrapper — it matches the Go engine conventions
# (family imgsz / embedded NMS / metadata). Full spec: docs/EXPORT_SPEC.md §6
python scripts/export_obb_onnx.py --weights yolo26m-obb.pt
python scripts/export_obb_onnx.py --verify-only ../models/yolo26/yolo26m-obb.onnx

# Eval & Benchmark (separate tool)
# raven-onnxruntime eval model.onnx ...
```

## Project Structure

```
raven-trainer/
├── raven_trainer/            # Python package
│   ├── dataset_pipeline/      # 7-layer Dataset Pipeline (60%)
│   │   ├── importers/          # Video/image import
│   │   ├── cleaners/           # Dedup + quality filter
│   │   ├── annotators/         # Annotation storage + converters
│   │   ├── versioning/         # Dataset version manager
│   │   ├── builders/           # Plugin workspace builders
│   │   ├── recipes/            # Dataset Recipe spec
│   │   ├── workspace.py        # Training workspace model
│   │   └── manifest.py         # Dataset manifest
│   ├── adapters/            # Backend adapter interface
│   ├── cli/                  # raven-train commands
│   ├── augment/             # Augmentation pipelines
│   ├── exporters/           # Export helpers
│   │   ...
├── projects/             # Business projects (config only)
├── configs/              # Configuration templates
├── docs/                  # Specifications
└── pyproject.toml
```

## Adapter Packages (separate pip packages)

```bash
pip install raven-ultralytics   # YOLO26 / YOLO11 (det/seg/pose/obb/cls)
pip install raven-mmdetection   # RT-DETR
pip install raven-anomalib      # Anomaly detection
pip install raven-rfdetr        # RF-DETR (det/seg)
```

## Documentation

| Document | Description |
|----------|-------------|
| [DATASET_PIPELINE_SPEC.md](./docs/dataset_pipeline/DATASET_PIPELINE_SPEC.md) | Dataset pipeline specification |
| [DATASET_SPEC.md](./docs/DATASET_SPEC.md) | Dataset & annotation specification |
| [MODEL_SPEC.md](./docs/MODEL_SPEC.md) | Model adapter specification |
| [TRAINING_SPEC.md](./docs/TRAINING_SPEC.md) | Training workflow specification |
| [AUGMENT_SPEC.md](./docs/AUGMENT_SPEC.md) | Data augmentation specification |
| [EXPORT_SPEC.md](./docs/EXPORT_SPEC.md) | Model export specification (incl. OBB §6: family conventions, `export_obb_onnx.py`, reference-data regeneration) |
| [CODING_SPEC.md](./docs/CODING_SPEC.md) | Coding standards & conventions |

## License

MIT
