"""PipelineProfile — 全生命周期管道配置。

Pipeline 是继 IngestionProfile 和 DatasetRecipe 之后的第三条对称管线：

  IngestionProfile        DatasetRecipe           PipelineProfile
       │                       │                       │
       ▼                       ▼                       ▼
  Raw Data ──→ Asset Dataset ──→ Training Workspace ──→ Trained Model
  storage/     datasets/         workspace/             projects/

保存 PipelineProfile，不保存处理结果。任何时候都可以用同一配置重新运行管道。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import R


@dataclass
class PipelineIngestionConfig:
    """Ingestion stage configuration."""
    enabled: bool = False
    raw_video_path: str = ""
    pipeline_stages: str = "motion,scene_change,burst,balance"
    profile_ref: str = ""  # optional reference to an IngestionProfile name


@dataclass
class PipelineDatasetConfig:
    """Dataset stage configuration."""
    name: str = ""
    task: str = "detection"
    source_dirs: List[str] = field(default_factory=list)


@dataclass
class PipelineAnnotationConfig:
    """Annotation stage configuration."""
    auto_annotate: bool = False
    model: str = "yolo26n.pt"
    conf_threshold: float = 0.25


@dataclass
class PipelineBuildConfig:
    """Build (Recipe → Workspace) stage configuration."""
    enabled: bool = True
    recipe_ref: str = "default"
    output_dir: str = ""
    force: bool = False


@dataclass
class PipelineTrainingConfig:
    """Training stage configuration."""
    enabled: bool = False
    model: str = "yolo26n"
    epochs: int = 100
    batch: int = 16
    device: str = "auto"


@dataclass
class PipelineProfile:
    """全生命周期管道配置文件。

    定义如何从原始数据一路走到训练出模型。
    保存配置，不保存处理结果。
    """
    name: str
    version: str = "1.0"
    description: str = ""
    created_at: str = ""

    # ── Stages ──
    ingestion: PipelineIngestionConfig = field(default_factory=PipelineIngestionConfig)
    dataset: PipelineDatasetConfig = field(default_factory=PipelineDatasetConfig)
    annotation: PipelineAnnotationConfig = field(default_factory=PipelineAnnotationConfig)
    build: PipelineBuildConfig = field(default_factory=PipelineBuildConfig)
    training: PipelineTrainingConfig = field(default_factory=PipelineTrainingConfig)

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at or datetime.now().isoformat(),
            "ingestion": {
                "enabled": self.ingestion.enabled,
                "raw_video_path": self.ingestion.raw_video_path,
                "pipeline_stages": self.ingestion.pipeline_stages,
                "profile_ref": self.ingestion.profile_ref,
            },
            "dataset": {
                "name": self.dataset.name,
                "task": self.dataset.task,
                "source_dirs": self.dataset.source_dirs,
            },
            "annotation": {
                "auto_annotate": self.annotation.auto_annotate,
                "model": self.annotation.model,
                "conf_threshold": self.annotation.conf_threshold,
            },
            "build": {
                "enabled": self.build.enabled,
                "recipe_ref": self.build.recipe_ref,
                "output_dir": self.build.output_dir,
                "force": self.build.force,
            },
            "training": {
                "enabled": self.training.enabled,
                "model": self.training.model,
                "epochs": self.training.epochs,
                "batch": self.training.batch,
                "device": self.training.device,
            },
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineProfile":
        ing = data.get("ingestion", {})
        ds = data.get("dataset", {})
        ann = data.get("annotation", {})
        bld = data.get("build", {})
        trn = data.get("training", {})

        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            ingestion=PipelineIngestionConfig(
                enabled=ing.get("enabled", False),
                raw_video_path=ing.get("raw_video_path", ""),
                pipeline_stages=ing.get("pipeline_stages", "motion,scene_change,burst,balance"),
                profile_ref=ing.get("profile_ref", ""),
            ),
            dataset=PipelineDatasetConfig(
                name=ds.get("name", data["name"]),
                task=ds.get("task", "detection"),
                source_dirs=ds.get("source_dirs", []),
            ),
            annotation=PipelineAnnotationConfig(
                auto_annotate=ann.get("auto_annotate", False),
                model=ann.get("model", "yolo26n.pt"),
                conf_threshold=ann.get("conf_threshold", 0.25),
            ),
            build=PipelineBuildConfig(
                enabled=bld.get("enabled", True),
                recipe_ref=bld.get("recipe_ref", "default"),
                output_dir=bld.get("output_dir", ""),
                force=bld.get("force", False),
            ),
            training=PipelineTrainingConfig(
                enabled=trn.get("enabled", False),
                model=trn.get("model", "yolo26n"),
                epochs=trn.get("epochs", 100),
                batch=trn.get("batch", 16),
                device=trn.get("device", "cuda:0"),
            ),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineProfile":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)


class PipelineProfileManager:
    """PipelineProfile 管理器——独立管理 data/pipeline/profiles/。"""

    def __init__(self):
        self.profiles_root = R.PIPELINE_PROFILES
        self.profiles_root.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[str]:
        if not self.profiles_root.exists():
            return []
        return sorted(f.stem for f in self.profiles_root.glob("*.yaml"))

    def get_profile(self, name: str) -> Optional[PipelineProfile]:
        path = self.profiles_root / f"{name}.yaml"
        if path.exists():
            return PipelineProfile.from_yaml(path)
        return None

    def save_profile(self, profile: PipelineProfile) -> Path:
        path = self.profiles_root / f"{profile.name}.yaml"
        profile.save(path)
        return path

    def delete_profile(self, name: str) -> None:
        path = self.profiles_root / f"{name}.yaml"
        if path.exists():
            path.unlink()

    def profile_to_request(self, profile: PipelineProfile) -> dict:
        """Convert a PipelineProfile to request params for _pipeline_thread."""
        return {
            "dataset_name": profile.dataset.name or profile.name,
            "task": profile.dataset.task,
            "source_dirs": list(profile.dataset.source_dirs),
            "raw_video_path": profile.ingestion.raw_video_path if profile.ingestion.enabled else None,
            "pipeline_stages": profile.ingestion.pipeline_stages,
            "auto_annotate": profile.annotation.auto_annotate,
            "model_name": profile.annotation.model,
            "conf_threshold": profile.annotation.conf_threshold,
            "build_workspace": profile.build.enabled,
            "output_dir": profile.build.output_dir or None,
            "force_build": profile.build.force,
            "start_training": profile.training.enabled,
            "training_model": profile.training.model,
            "training_epochs": profile.training.epochs,
            "training_batch": profile.training.batch,
        }
