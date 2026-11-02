"""IngestionProfile — 数据摄入配置（等同于 Ingestion 的 Recipe）。

定义如何从 Raw Data 构建 Asset Dataset：
  - 使用哪些 Raw Sources（视频/图片目录）
  - 解码配置（interval / fps / keyframe 抽帧模式）
  - 采样策略（motion, scene_change, object, burst）
  - 质量过滤参数（blur, black, overexposed, resolution）
  - 去重配置（dhash, phash, ssim）

保存 IngestionProfile，不保存处理结果。
任何时候都可以用同一配置重新 Ingestion。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import os

from ..config import validate_name
from .sampling.core import DecodeConfig


def _filter_kwargs(dc_type: type, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only keyword arguments accepted by ``dc_type``'s dataclass fields.

    Profiles are user-editable YAML; unknown or stale keys must be ignored
    instead of raising ``TypeError`` in the middle of an ingestion run.
    """
    import dataclasses

    allowed = {f.name for f in dataclasses.fields(dc_type)}
    return {k: v for k, v in (values or {}).items() if k in allowed}


@dataclass
class SourceRef:
    """Raw source reference."""
    name: str                       # short name, e.g. "camera01"
    path: str                        # path relative to storage/, e.g. "videos/workshop_a"
    source_type: str = "video"       # "video" | "image_dir"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)


@dataclass
class IngestionProfile:
    """数据摄入配置文件——定义如何从 Raw Data 构建 Asset Dataset。

    类似 DatasetRecipe 但作用于 Raw → AssetDataset 阶段。
    保存 profile，不保存处理结果。任何时候都可以重新 Ingestion。
    """
    name: str
    version: str = "1.0"
    description: str = ""
    created_at: str = ""

    # Raw sources
    sources: List[SourceRef] = field(default_factory=list)

    # ── Decode config (how to read frames from video) ──
    decode: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "interval",          # interval | fps | keyframe
        "every_n_frames": 30,
        "target_fps": 0.0,
        "jpeg_quality": 80,
    })

    # Sampling pipeline
    pipeline: List[str] = field(default_factory=lambda: [
        "motion", "scene_change", "burst", "balance"
    ])
    sample_count: int = 500

    # Sampling config
    sampling: Dict[str, Any] = field(default_factory=lambda: {
        "interval": {"interval": 30},
        "motion": {"motion_area_threshold": 0.5},
        "scene_change": {"method": "histogram", "threshold": 0.3},
        "burst": {"before": 5, "after": 10},
        "balance": {"max_per_camera": 5000},
    })

    # Quality config
    quality: Dict[str, Any] = field(default_factory=lambda: {
        "blur_laplacian_threshold": 100.0,
        "black_mean_threshold": 5.0,
        "overexposed_mean_threshold": 250.0,
        "overexposed_pixel_ratio": 0.5,
        "min_width": 320,
        "min_height": 240,
    })

    # Dedup config
    dedup: Dict[str, Any] = field(default_factory=lambda: {
        "method": "dhash",
        "hash_size": 8,
        "hash_distance": 5,
        "ssim_threshold": 0.98,
    })

    # Auto-register
    auto_register: bool = True
    target_dataset: str = ""         # Asset dataset name to register into

    extra: Dict[str, Any] = field(default_factory=dict)

    def get_decode_config(self) -> DecodeConfig:
        """Build a DecodeConfig from this profile's decode dict."""
        d = self.decode
        return DecodeConfig(
            mode=d.get("mode", "interval"),
            every_n_frames=d.get("every_n_frames", 30),
            target_fps=d.get("target_fps", 0.0),
            jpeg_quality=d.get("jpeg_quality", 80),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestionProfile":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionProfile":
        sources = [SourceRef(**_filter_kwargs(SourceRef, s)) for s in data.get("sources", [])]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            sources=sources,
            decode=data.get("decode", {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80}),
            pipeline=data.get("pipeline", ["motion", "scene_change", "burst", "balance"]),
            sample_count=data.get("sample_count", 500),
            sampling=data.get("sampling", {}),
            quality=data.get("quality", {}),
            dedup=data.get("dedup", {}),
            auto_register=data.get("auto_register", True),
            target_dataset=data.get("target_dataset", ""),
            extra=data.get("extra", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at or datetime.now().isoformat(),
            "sources": [{"name": s.name, "path": s.path, "source_type": s.source_type,
                         "include": s.include, "exclude": s.exclude} for s in self.sources],
            "decode": self.decode,
            "pipeline": self.pipeline,
            "sample_count": self.sample_count,
            "sampling": self.sampling,
            "quality": self.quality,
            "dedup": self.dedup,
            "auto_register": self.auto_register,
            "target_dataset": self.target_dataset,
            "extra": self.extra,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)


class IngestionProfileManager:
    """IngestionProfile 管理器——独立管理 data/ingestion/profiles/。"""

    def __init__(self, profiles_root: Optional[Path] = None):
        from ..config import R as _R
        self.profiles_root = profiles_root or _R.INGESTION_PROFILES
        self.profiles_root.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[str]:
        if not self.profiles_root.exists():
            return []
        return sorted(
            f.stem for f in self.profiles_root.glob("*.yaml")
        )

    def _profile_path(self, name: str) -> Path:
        """Resolve a profile YAML path, rejecting unsafe names."""
        return self.profiles_root / f"{validate_name(name, 'profile name')}.yaml"

    def get_profile(self, name: str) -> Optional[IngestionProfile]:
        path = self._profile_path(name)
        if path.exists():
            return IngestionProfile.from_yaml(path)
        return None

    def save_profile(self, profile: IngestionProfile) -> Path:
        path = self._profile_path(profile.name)
        profile.save(path)
        return path

    def delete_profile(self, name: str) -> None:
        path = self._profile_path(name)
        if path.exists():
            path.unlink()

    def run_profile(self, name: str, output_dir: Optional[Path] = None) -> Any:
        """执行一个 IngestionProfile，返回 IngestionReport。"""
        profile = self.get_profile(name)
        if profile is None:
            raise FileNotFoundError(f"IngestionProfile '{name}' not found")

        from .pipeline import IngestionPipeline
        from .core import QualityFilterConfig, DedupConfig

        # Unknown keys in a hand-edited YAML must not crash ingestion: keep only
        # the fields the dataclasses actually accept.
        quality = QualityFilterConfig(**_filter_kwargs(QualityFilterConfig, profile.quality))
        dedup = DedupConfig(**_filter_kwargs(DedupConfig, profile.dedup))

        # Resolve raw root from first source's parent
        from ..config import R as _R
        raw_root = _R.STORAGE
        if profile.sources:
            # Compute the common parent of all source paths
            # so RawImporter.scan() can find all subdirectories/files
            source_parents = [Path(s.path).parent for s in profile.sources]
            try:
                common = (
                    Path(os.path.commonpath([str(p) for p in source_parents]))
                    if len(source_parents) > 1
                    else source_parents[0]
                )
            except ValueError:
                # Different drives (Windows) — fall back to the storage root.
                common = Path(".")
            candidate = (_R.STORAGE / common).resolve()
            storage_root = _R.STORAGE.resolve()
            # Never let a profile point outside storage/.
            if candidate == storage_root or storage_root in candidate.parents:
                if candidate.exists():
                    raw_root = candidate
            else:
                raw_root = _R.STORAGE

        pipeline = IngestionPipeline(
            raw_root=raw_root,
            decode_config=profile.get_decode_config(),
            sampling_pipeline=profile.pipeline,
            sampler_config=profile.sampling,
            sample_count=profile.sample_count,
            quality=quality,
            dedup=dedup,
            auto_register=profile.auto_register,
            target_dataset=profile.target_dataset or profile.name,
        )

        return pipeline.run(output_dir=output_dir or _R.ingestion_run_dir(f"profile_{name}"))
