"""Base adapter interface.

Adapters wrap upstream model frameworks (Ultralytics, MMDetection, etc.)
and expose a uniform interface to Raven Trainer.

Methods:
  - train:     run the training loop
  - evaluate:  evaluate a trained model on a validation set (returns metrics dict)
  - export:    export trained model to ONNX / deployment format

Each method can be implemented independently per backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class ModelTask(str, Enum):
    DETECTION = "detection"
    POSE = "pose"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ANOMALY = "anomaly"
    OBB = "obb"


@dataclass
class ModelInfo:
    """Metadata exposed by an adapter for listing & discovery."""
    backend: str
    task: ModelTask
    framework: str
    supported_models: List[str] = field(default_factory=list)
    description: str = ""


class ModelAdapter(ABC):
    """Abstract base for all backend adapters."""

    backend_name: str = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model = None

    @classmethod
    @abstractmethod
    def info(cls) -> ModelInfo:
        ...

    @abstractmethod
    def train(self, dataset_cfg: str | Path, **kwargs) -> Any:
        """Run training loop. Returns the trained model or training results."""
        ...

    def evaluate(self, dataset_cfg: str | Path, **kwargs) -> Dict[str, float]:
        """Evaluate a trained model on a dataset. Returns metrics dict.

        Optional method. If not overridden, returns empty dict.
        Adapters should override this to provide framework-specific evaluation.
        """
        return {}

    @abstractmethod
    def export(self, output_path: str | Path, **kwargs) -> Dict[str, Any]:
        """Export trained model to ONNX or other deployment format.

        Returns a dict with at least ``onnx_path`` (str) and optional
        verification metadata. Subclasses MUST return a dict, not a Path.
        """
        ...

    def artifact_paths(self, run_dir: str) -> Dict[str, str]:
        """Locate framework-specific artifacts in a training run directory.

        Optional helper so CLI code does not have to assume the Ultralytics
        ``runs/<run>/weights/best.pt`` layout. Returns a dict with optional
        keys, e.g. ``best_model`` / ``last_ema`` / ``last_checkpoint`` /
        ``metrics_csv``. Base implementation returns an empty dict.
        """
        return {}

    @property
    def model(self) -> Any:
        return self._model

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError
