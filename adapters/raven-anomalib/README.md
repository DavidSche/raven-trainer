# raven-anomalib

**Raven Trainer — Anomalib (anomaly detection) backend adapter.**

## Install

```bash
pip install raven-anomalib
```

## Usage

```bash
raven-train new --name defect_detection --task anomaly --backend anomalib --model patchcore
raven-train train projects/defect_detection/project.yaml
```

## Supported Models

- PatchCore
- FastFlow
- EfficientAD
- PaDiM

v1 production validation targets are **EfficientAD-S/256** and **PaDiM-R18/256**. Other variants are contract-compatible and require their own export fixtures, thresholds, qualification data, and benchmark evidence.

## Raven RMP Export / Inference Loop

1. Train or fit the selected anomalib model with normal images for model fitting.
2. Keep calibration images separate from fitting images. Run threshold calibration with only validation data and representative anomaly samples.
3. Export through the Raven RMP path (`manifest.json` + ONNX + calibrated `score_threshold` + threshold report + export profile). Do not bypass RMP or embed an ad hoc threshold in the manifest.
4. Load the resulting RMP package in raven-go as an `anomaly` task model.
5. Validate deployment identity before production: model artifact hash, threshold version, decision policy version, ROI/camera binding, input size, score semantics, and export profile.
6. Run scenario qualification on held-out data. Do not promote a model based only on fitting-set AUROC or a golden numerical PASS.

The local RMP fixtures and numerical harness live in `raven-onnxruntime`; the runtime contract and qualification evidence requirements are documented in `raven-go/doc/raven-onnxruntime-efficientad-padim-integration-plan.md`.
