"""Base dataset abstractions for the Raven Trainer platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class TaskType(str, Enum):
    """Supported computer vision tasks."""

    DETECTION = "detection"
    POSE = "pose"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ANOMALY = "anomaly"
    OBB = "obb"


class AnnotationFormat(str, Enum):
    """Supported annotation formats."""

    COCO = "coco"
    YOLO = "yolo"
    CVAT = "cvat"
    LABEL_STUDIO = "label_studio"


@dataclass
class Annotation:
    """A single annotation (bounding box, polygon, keypoint, etc.)."""

    label_id: int
    label_name: str
    bbox: Optional[List[float]] = None  # [x, y, w, h] in normalized coords
    segmentation: Optional[List[float]] = None  # polygon points
    keypoints: Optional[List[float]] = None  # [x1, y1, v1, x2, y2, v2, ...]
    area: Optional[float] = None
    iscrowd: bool = False
    ignore: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_yolo(self) -> str:
        """Convert to YOLO label format string."""
        if self.bbox is None:
            raise ValueError("bbox required for YOLO format")
        x, y, w, h = self.bbox
        return f"{self.label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    def to_yolo_obb(self) -> str:
        """Convert to YOLO OBB label format: class x1 y1 x2 y2 x3 y3 x4 y4.

        Uses the 4 corner points from ``segmentation`` (8 floats) if available,
        otherwise derives corners from the axis-aligned ``bbox``.
        """
        if self.segmentation and len(self.segmentation) >= 8:
            pts = self.segmentation[:8]
            return f"{self.label_id} " + " ".join(f"{p:.6f}" for p in pts)
        if self.bbox is None:
            raise ValueError("bbox or segmentation required for OBB format")
        x, y, w, h = self.bbox
        # Derive 4 corners from axis-aligned bbox
        corners = [x, y, x + w, y, x + w, y + h, x, y + h]
        return f"{self.label_id} " + " ".join(f"{p:.6f}" for p in corners)


@dataclass
class Sample:
    """A single dataset sample (image + annotations)."""

    image_path: Path
    image_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    annotations: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset configuration loaded from dataset.yaml."""

    path: str
    train: str = "train"
    val: str = "val"
    test: Optional[str] = None
    names: Dict[int, str] = field(default_factory=dict)
    task: TaskType = TaskType.DETECTION
    format: AnnotationFormat = AnnotationFormat.YOLO
    hard_negative_ratio: float = 0.0
    split_method: str = "video_based"  # 'random' | 'video_based' | 'temporal'

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        """Load dataset configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Coerce string values to their enum types so that to_dict() can call
        # .value on them without AttributeError. Also ignore unknown fields so
        # YAML files with extra metadata (author, notes, etc.) don't crash.
        if isinstance(data.get("task"), str):
            data["task"] = TaskType(data["task"])
        if isinstance(data.get("format"), str):
            data["format"] = AnnotationFormat(data["format"])
        # Filter to known dataclass fields to tolerate extra YAML keys
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Save dataset configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "names": self.names,
            "task": self.task.value,
            "format": self.format.value,
            "hard_negative_ratio": self.hard_negative_ratio,
            "split_method": self.split_method,
        }


class BaseDataset:
    """Base class for all datasets in the Raven Trainer platform."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.samples: List[Sample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def load(self) -> None:
        """Load dataset samples. Override in subclasses."""
        raise NotImplementedError

    def filter(self, **kwargs) -> "BaseDataset":
        """Filter dataset samples by metadata."""
        raise NotImplementedError

    def split(self, ratios: List[float], method: str = "random") -> List["BaseDataset"]:
        """Split dataset into train/val/test subsets."""
        raise NotImplementedError
