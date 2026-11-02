"""raven-train new — scaffold a new training project.

Usage:
    raven-train new --name person_intrusion --task detection \
        --backend ultralytics --model yolo26n

Models (--model): yolo26n/s/m/l/x, yolo11n/s/m/l/x, rtdetr-l
    (+ -seg / -pose / -obb / -cls suffixes for YOLO26/11); rf-detr-* for rfdetr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


from typing import Any, Dict, Optional

import click
import yaml

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent.parent.parent
PROJECTS_DIR = Path(__file__).resolve().parent.parent.parent / "projects"

SUPPORTED_TASKS = {
    "detection": {"backend": ["ultralytics", "mmdetection", "rfdetr"], "models": ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "rtdetr-l", "rf-detr-nano", "rf-detr-small", "rf-detr-medium", "rf-detr-large"]},
    "pose": {"backend": ["ultralytics"], "models": ["yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose", "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose"]},
    "segmentation": {"backend": ["ultralytics", "rfdetr"], "models": ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg", "rf-detr-seg-nano", "rf-detr-seg-small", "rf-detr-seg-medium", "rf-detr-seg-large"]},
    "classification": {"backend": ["ultralytics"], "models": ["yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls", "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls"]},
    "anomaly": {"backend": ["anomalib"], "models": ["patchcore", "fastflow", "efficientad"]},
    "obb": {"backend": ["ultralytics"], "models": ["yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb", "yolo26x-obb", "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb"]},
}

SUPPORTED_DATASETS = ["cvat", "label_studio", "local"]
SUPPORTED_BACKENDS = ["ultralytics", "mmdetection", "anomalib", "rfdetr"]


# -------------------------------------------------------------------
# Project manifest
# -------------------------------------------------------------------


@dataclass
class ProjectManifest:
    """project.yaml — single source of truth for a training project."""

    name: str
    version: str
    task: str
    created: str = ""
    backend: Dict[str, Any] = field(default_factory=lambda: {"name": "ultralytics", "model": "yolo26n"})
    dataset: Dict[str, Any] = field(default_factory=lambda: {"source": "cvat", "version": "", "ref": ""})
    export: Dict[str, Any] = field(default_factory=lambda: {"format": "onnx", "opset": 17})
    trainer: Dict[str, Any] = field(default_factory=lambda: {"version": "0.2.0"})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "created": self.created or datetime.now().strftime("%Y.%m.%d"),
            "backend": self.backend,
            "dataset": self.dataset,
            "export": self.export,
            "trainer": self.trainer,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectManifest":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# -------------------------------------------------------------------
# Scaffolder
# -------------------------------------------------------------------


class ProjectScaffolder:
    """Create a Raven Trainer project.

    The generated project contains ONLY configuration and references.
    There is no training code — that lives in raven-trainer itself.
    """

    def __init__(
        self,
        name: str,
        task: str,
        backend: str,
        model: str,
        dataset_source: str,
        template: Optional[str] = None,
    ):
        self.name = name
        self.task = task
        self.backend = backend
        self.model = model
        self.dataset_source = dataset_source
        self.template = template or "industrial"
        self.project_dir = PROJECTS_DIR / name

        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Choose from: {list(SUPPORTED_TASKS)}")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Choose from: {SUPPORTED_BACKENDS}")
        if dataset_source not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset source '{dataset_source}'. Choose from: {SUPPORTED_DATASETS}")

    def run(self) -> Path:
        if self.project_dir.exists():
            raise FileExistsError(f"Project '{self.name}' already exists at {self.project_dir}")

        if self.template and self.template != 'industrial':
            click.echo(f"Warning: template '{self.template}' not implemented, using default")

        self._create_directories()
        self._create_project_manifest()
        self._create_dataset_config()
        self._create_train_config()
        self._create_readme()

        click.echo(f"[Raven Trainer] Project created: {self.project_dir}")
        self._print_summary()
        return self.project_dir

    def _create_directories(self) -> None:
        for d in ["configs", "datasets", "experiments", "outputs"]:
            (self.project_dir / d).mkdir(parents=True, exist_ok=True)
        (self.project_dir / "experiments").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models").mkdir(parents=True, exist_ok=True)

    def _create_project_manifest(self) -> None:
        manifest = ProjectManifest(
            name=self.name,
            version="1.0",
            task=self.task,
            backend={"name": self.backend, "model": self.model},
            dataset={"source": self.dataset_source, "version": "", "ref": ""},
        )
        self._write_yaml(self.project_dir / "project.yaml", manifest.to_dict())

    def _create_dataset_config(self) -> None:
        config = {
            "path": str(self.project_dir / "datasets"),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": {0: "object"},
            "task": self.task,
            "format": "yolo",
            "hard_negative_ratio": 0.0,
            "split_method": "video_based",
        }
        self._write_yaml(self.project_dir / "dataset.yaml", config)

    def _create_train_config(self) -> None:
        config = {
            "epochs": 300,
            "batch": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "amp": True,
            "patience": 50,
            "close_mosaic": 10,
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
            "export": {
                "format": "onnx",
                "opset": 17,
                "dynamic": True,
                "half": False,
                "metadata": {
                    "model_name": self.name,
                    "model_version": "1.0",
                },
            },
        }
        self._write_yaml(self.project_dir / "train.yaml", config)

    def _create_readme(self) -> None:
        readme = self.project_dir / "README.md"
        content = (
            f"# {self.name}\n\n"
            f"## Overview\n\n"
            f"- **Task:** {self.task}\n"
            f"- **Backend:** {self.backend}\n"
            f"- **Model:** {self.model}\n"
            f"- **Dataset Source:** {self.dataset_source}\n"
            f"- **Created:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"## Quick Start\n\n"
            f"```bash\n"
            f"# Train\n"
            f"raven-train train project.yaml\n\n"
            f"# Export (eval/benchmark via raven-onnxruntime)\n"
            f"raven-train export project.yaml\n"
            f"```\n"
        )
        readme.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_yaml(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _print_summary(self) -> None:
        click.echo(f"  +-- project.yaml     (manifest — backend: {self.backend})")
        click.echo("  +-- dataset.yaml     (dataset config)")
        click.echo("  +-- train.yaml       (train + export config)")
        click.echo("  +-- configs/         (extra configs)")
        click.echo("  +-- datasets/        (data references)")
        click.echo("  +-- experiments/     (experiment logs)")
        click.echo("  +-- outputs/         (model outputs)")
        click.echo("  +-- README.md")


# -------------------------------------------------------------------
# CLI command
# -------------------------------------------------------------------


@click.command(name="new")
@click.option("--name", "-n", required=True, help="Project name (snake_case)")
@click.option("--task", "-t", required=True, help=f"Task: {', '.join(SUPPORTED_TASKS)}")
@click.option("--backend", "-b", default="ultralytics", help=f"Model backend: {', '.join(SUPPORTED_BACKENDS)}")
@click.option("--model", "-m", default="yolo26n", help="Model name (e.g. yolo26n, yolo11m, rtdetr-l)")
@click.option("--dataset", "-d", default="cvat", help=f"Dataset source: {', '.join(SUPPORTED_DATASETS)}")
@click.option("--template", help="Project template (optional)")
def cmd_new(name: str, task: str, backend: str, model: str, dataset: str, template: str | None):
    """Create a new training project."""
    scaffolder = ProjectScaffolder(
        name=name,
        task=task,
        backend=backend,
        model=model,
        dataset_source=dataset,
        template=template,
    )
    scaffolder.run()
