"""Raven Trainer — MMDetection (RT-DETR) backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raven_trainer.adapters.base import ModelAdapter, ModelInfo, ModelTask


class RavenMMDetectionAdapter(ModelAdapter):
    """Adapter for MMDetection models (RT-DETR, etc.)."""

    backend_name = "mmdetection"

    @classmethod
    def info(cls) -> ModelInfo:
        return ModelInfo(
            backend="mmdetection",
            task=ModelTask.DETECTION,
            framework="mmdetection",
            supported_models=["rtdetr-l"],
            description="MMDetection models",
        )

    def train(self, dataset_cfg: str | Path, **kwargs) -> Any:
        raise NotImplementedError("MMDetection training not yet implemented")

    def export(self, output_path: str | Path, **kwargs) -> Path:
        raise NotImplementedError("MMDetection export not yet implemented")
