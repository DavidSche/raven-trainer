# raven-ultralytics

**Raven Trainer — Ultralytics (YOLO) backend adapter.**

Installs the Ultralytics framework and registers it with Raven Trainer.

## Install

```bash
pip install raven-ultralytics
```

## Usage

```bash
# Create a YOLO project
raven-train new --name person_intrusion --task detection --backend ultralytics --model yolo11m

# Train (requires raven-ultralytics installed)
raven-train train projects/person_intrusion/project.yaml
```

## Supported Models

- YOLO11 n/s/m/l/x (detection, pose, segmentation, classification)
