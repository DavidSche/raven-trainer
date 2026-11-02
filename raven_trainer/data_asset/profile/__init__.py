"""Build Profile — 训练构建配置，与 Recipe 解耦。

Recipe 负责：数据怎么组成（哪些来源、什么比例、如何 split）。
Profile 负责：训练怎么构建（目标格式、过滤规则、数据增强）。

一个 Dataset + 一个 Recipe = 一个 Workspace（数据的逻辑组成）。
一个 Workspace + 一个 Profile = 一个 Training Workspace（训练物化格式）。

示例：
  profile_detection.yaml:
    task: detection
    target_format: yolo
    include_classes: [person, helmet]
    exclude_classes: []
    filter:
      min_blur_score: 30
      max_instances: 500
    augment:
      copy_paste: true
      mosaic: true
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BuildProfile:
    """训练构建配置。

    与 Recipe 的区别：
      Recipe  = 数据从哪里来、选哪些样本
      Profile = 数据如何格式化、目标是什么
    """

    name: str
    version: str = "1.0"
    task: str = "detection"               # detection, pose, seg
    target_format: str = "yolo"           # yolo, coco, cvat

    # 类别过滤
    include_classes: List[int] = field(default_factory=list)
    exclude_classes: List[int] = field(default_factory=list)

    # 过滤规则
    filter: Dict[str, Any] = field(default_factory=lambda: {
        "min_blur_score": 30,
        "min_quality_score": 20,
        "max_images_per_class": 5000,
        "remove_empty_labels": True,
    })

    # 数据增强
    augment: Dict[str, Any] = field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.3,
    })

    # 输出配置
    output: Dict[str, Any] = field(default_factory=lambda: {
        "val_size": 0.2,
        "test_size": 0.0,
        "link": True,
    })

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "target_format": self.target_format,
            "include_classes": self.include_classes,
            "exclude_classes": self.exclude_classes,
            "filter": self.filter,
            "augment": self.augment,
            "output": self.output,
            "extra": self.extra,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "BuildProfile":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"BuildProfile must be a mapping, got {type(data).__name__}")
        # Filter to known dataclass fields to tolerate extra YAML keys
        known = {f.name for f in dataclass_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def detection_default(cls, name: str = "default") -> "BuildProfile":
        return cls(name=name, task="detection")

    @classmethod
    def pose_default(cls, name: str = "pose") -> "BuildProfile":
        return cls(name=name, task="pose", target_format="coco")

    @classmethod
    def seg_default(cls, name: str = "seg") -> "BuildProfile":
        return cls(name=name, task="segmentation", target_format="coco")
