# Raven Trainer

**English** | [简体中文](./README.zh-CN.md)

Industrial-vision **data asset & model training platform**: turn raw video into versioned
datasets, materialize reproducible training workspaces from recipes, train with pluggable
backends, and export deployment-ready ONNX + metadata.

Raven Trainer is the training counterpart of **Raven-Go** (runtime/engine). It ships a
FastAPI backend with a React console, a Click CLI (`raven-train`), and a `uv` workspace of
backend adapters.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-3776AB" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-75%20passed-brightgreen" />
  <img alt="mypy" src="https://img.shields.io/badge/mypy-0%20errors-brightgreen" />
</p>

---

## Why

A dataset is code, not files. The **Dataset Recipe** is the source of truth: a declarative
description of *how* a dataset is built (sources, sampling, quality gates, annotation,
split). Keep the recipe and regenerate the data — the same recipe always yields the same
dataset, and every artifact stays traceable.

## Three-layer storage model

```
storage/      raw, immutable        imported video/images, never modified in place
   ↓
datasets/     versioned assets      <dataset>/versions/vN/{manifest.jsonl, snapshot.yaml, annotations/}
   ↓
workspace/    materialized recipe   train/ val/ test/ + data.yaml, ready for a trainer
```

`recipes/` sits outside the three layers: it holds the build rules that connect
`datasets/` to `workspace/`.

Versions are **append-only**: a version directory appears atomically, an explicit tag
collision is an error, and no module may hand-write a manifest — every write goes through
`VersionManager`.

## Pipeline

```
Raw video → Ingest (sample / quality gate / dedup) → Register → Version
          → Recipe → Workspace (split + data.yaml) → Train → Export (ONNX + metadata)
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `raven_trainer/ingestion` | frame sampling (interval / motion / burst / object-driven / scene-change), blur / black / over-exposure / small-object rejection, dHash + SSIM dedup |
| Register & version | `raven_trainer/data_asset` | registry, manifests, atomic version snapshots, tags, lineage, SQLite metadata, feedback merge |
| Recipe → workspace | `data_asset/recipe`, `data_asset/workspace` | declarative build recipe, deterministic train/val/test split, backend-specific layout |
| Train | `raven_trainer/adapters`, `experiments/` | backend adapters (Ultralytics / MMDetection / Anomalib / RF-DETR), experiment records, live log streaming |
| Export | `raven_trainer/exporters`, `scripts/export_obb_onnx.py` | ONNX export with family conventions (NMS, dynamic shapes) plus a metadata sidecar |

Annotations are stored in **CVAT XML / COCO JSON** (lossless and tool-friendly), never as
YOLO txt; YOLO/COCO layouts are produced at workspace build time.

## Quick start

Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/DavidSche/raven-trainer.git
cd raven-trainer
uv sync                      # core only (no torch)
uv sync --extra ultralytics  # add one backend: ultralytics | rfdetr | anomalib | mmdetection
uv sync --extra all          # all backends

# 1. scaffold a project (config only, no code)
uv run raven-train new --name person_intrusion --task detection --model yolo26n

# 2. ingest raw video into a versioned dataset
uv run raven-train ingest run --help          # sampling / quality / dedup / scene options
uv run raven-train dataset create person_v1
uv run raven-train dataset version person_v1 --create --source data/ingestion/runs/<run>/kept

# 3. build a training workspace from a recipe
uv run raven-train dataset recipe person_recipe --create
uv run raven-train dataset recipe person_recipe --build

# 4. train and export
uv run raven-train train projects/person_intrusion/train.yaml --epochs 100
uv run raven-train export projects/person_intrusion/project.yaml --register

# 5. web console (FastAPI + React, http://127.0.0.1:8081)
uv run raven-train ui --open
```

All state lives under `data/` in the repository root; override with `RAVEN_DATA=/path`.

### CLI map

| Command | Purpose |
|---|---|
| `raven-train new` | scaffold a project (`project.yaml`, `train.yaml`, `dataset.yaml`) |
| `raven-train ingest` | ingestion pipeline: `scan` · `run` · `profile` · `report` |
| `raven-train dataset` | data assets: `create` `list` `show` `version` `recipe` `build` `annotate` `task` `doctor` `lineage` `stats` `analyze` `tag` `misdata` |
| `raven-train train` | train through a backend adapter and record the experiment |
| `raven-train eval` | evaluate a trained model on a dataset version |
| `raven-train export` | export ONNX (+ metadata) for Raven-Go / raven-onnxruntime |
| `raven-train job` | inspect the unified job store: `list` · `show` · `log` |
| `raven-train ui` | serve the FastAPI backend and the React console |

### HTTP API

The backend mounts `/api/v1` (datasets, versions, annotations, ingestion profiles, training
jobs, models, pipeline, feedback, reports, tasks, storage, system) and `/api/health`.
Interactive OpenAPI docs are served at `/docs` while the server is running.

Long-running work (ingestion, training, export, pipeline) is tracked in the job store and
streams progress over WebSocket; endpoints that touch disk or CPU are synchronous so they
run in the server's thread pool instead of blocking the event loop.

## Backend adapters

Backends are separate distributions in a `uv` workspace, so the core install stays light
(no torch). Each adapter declares its own heavy dependencies.

| Package | Backend | Tasks |
|---|---|---|
| `raven-ultralytics` | Ultralytics (YOLO26 / YOLO11) | detect, segment, pose, obb, classify |
| `raven-mmdetection` | MMDetection (RT-DETR) | detect |
| `raven-anomalib` | Anomalib | anomaly detection |
| `raven-rfdetr` | RF-DETR | detect, segment |

Adding a backend means implementing the contract in `raven_trainer/adapters/base.py`;
the trainer, exporter and UI stay unchanged.

## Project layout

```
raven_trainer/
├── config.py        # R: canonical path helpers + validate_name (the single path gate)
├── data_asset/      # registry, version, recipe, workspace, metadata, tag, lineage,
│                    #   annotation (task), feedback, job, doctor, misdata, split
├── ingestion/       # profile, pipeline, sampling, quality, dedup, scene, report
├── datasets/        # dataset config + converters (CVAT / Label Studio), statistics
├── adapters/        # adapter contract + builtin backends
├── experiments/     # experiment records (exp_NNNN) and artifacts
├── exporters/       # ONNX export helpers
├── api/             # FastAPI app + routers
└── cli/             # raven-train command tree
adapters/            # workspace packages: raven-{ultralytics,mmdetection,anomalib,rfdetr}
ui_src/              # React 19 + Vite + TypeScript console
scripts/ · tools/    # operational scripts (import, export, dataset checks)
tests/               # pytest suite (safety, API smoke, CLI)
```

## Development

```bash
uv run pytest -q                     # 75 tests
uv run mypy raven_trainer            # 0 errors
uv run pylint raven_trainer          # 8.77 / 10
python -m compileall -q raven_trainer
```

`tests/` has no `__init__.py` on purpose, so lint test files explicitly:
`uv run pylint tests/test_safety.py tests/test_api_smoke.py`.

Rules of the road (see `AGENTS.md`): never modify `data/`, `.venv/`, `__pycache__/`,
`*.egg-info/`, `dist/`, `build/` or model weights; every user-supplied name goes through
`validate_name()`; every client-supplied path must resolve inside `PROJECT_ROOT` or `DATA`.

## Documentation

Detailed specifications (dataset, model, training, augmentation, export, coding) are kept
with the internal design documents and are **not published in this repository yet**. The
most useful in-repo references today are `AGENTS.md`, `raven_trainer/config.py` (path
contract), `raven_trainer/adapters/base.py` (backend contract) and the OpenAPI schema
served at `/docs`.

## License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](./LICENSE).
You may not use this project except in compliance with the License.
