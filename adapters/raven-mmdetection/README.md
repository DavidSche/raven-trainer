# raven-mmdetection

**Raven Trainer — MMDetection (RT-DETR) backend adapter.**

## Install

```bash
pip install raven-mmdetection
```

## Usage

```bash
raven-train new --name smoke_detection --task detection --backend mmdetection --model rtdetr-l
raven-train train projects/smoke_detection/project.yaml
```

## Supported Models

- RT-DETR-L
