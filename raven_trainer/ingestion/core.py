"""Core data models for the Ingestion Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class RawSourceMetadata:
    """Metadata for a raw video/image source.
    Import only builds this index -- it does NOT extract frames.
    """

    source_name: str
    source_type: str          # "video" | "image_dir"
    path: str
    fps: Optional[float] = None
    resolution: Optional[str] = None
    total_frames: Optional[int] = None
    duration_sec: Optional[float] = None
    file_size_mb: Optional[float] = None
    recording_time: Optional[str] = None
    sha256: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    imported_at: str = ""

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "path": self.path,
            "fps": self.fps,
            "resolution": self.resolution,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "file_size_mb": self.file_size_mb,
            "recording_time": self.recording_time,
            "sha256": self.sha256,
            "extra": self.extra,
            "imported_at": self.imported_at or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawSourceMetadata":
        return cls(**d)


@dataclass
class RawSource:
    """A single raw source file/folder with its metadata."""
    path: Path
    metadata: Optional[RawSourceMetadata] = None


@dataclass
class ImportedSample:
    """A single sampled frame after ingestion."""
    source_name: str
    frame_path: Path
    frame_index: int
    timestamp_sec: float = 0.0
    width: int = 0
    height: int = 0
    motion_score: float = 0.0
    scene_label: str = ""
    quality_score: float = 1.0
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityFilterConfig:
    """Thresholds for quality checks."""
    blur_laplacian_threshold: float = 100.0
    black_mean_threshold: float = 5.0
    overexposed_mean_threshold: float = 250.0
    overexposed_pixel_ratio: float = 0.5
    min_width: int = 320
    min_height: int = 240



@dataclass
class DedupConfig:
    """Configuration for perceptual deduplication."""
    method: str = "dhash"
    hash_size: int = 8
    hash_distance: int = 5
    ssim_threshold: float = 0.98


class SamplingStrategy(str, Enum):
    ADAPTIVE = "adaptive"
    EVENT_BASED = "event_based"
    INTERVAL = "interval"


@dataclass
class SamplingConfig:
    strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE
    interval: int = 30
    adaptive_min_fps: float = 0.2
    adaptive_max_fps: float = 5.0
    motion_threshold: float = 30.0
    event_cooldown_frames: int = 150
    event_empty_interval: int = 60


@dataclass
class IngestionReport:
    """Full report for a single ingestion run."""
    total_sources: int = 0
    total_raw_frames: int = 0
    total_sampled: int = 0
    total_kept_after_quality: int = 0
    total_after_dedup: int = 0
    quality_removed_blurry: int = 0
    quality_removed_black: int = 0
    quality_removed_overexposed: int = 0
    quality_removed_small: int = 0
    dedup_removed: int = 0
    scene_distribution: Dict[str, int] = field(default_factory=dict)
    resolution_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    samples: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Sources: {self.total_sources}",
            f"Raw frames: {self.total_raw_frames}",
            f"After sampling: {self.total_sampled}",
            f"After quality check: {self.total_kept_after_quality}",
            f"  - removed blurry: {self.quality_removed_blurry}",
            f"  - removed black: {self.quality_removed_black}",
            f"  - removed overexposed: {self.quality_removed_overexposed}",
            f"  - removed too small: {self.quality_removed_small}",
            f"After dedup: {self.total_after_dedup}",
            f"  - removed duplicates: {self.dedup_removed}",
        ]
        if self.scene_distribution:
            lines.append(f"Scenes: {self.scene_distribution}")
        return "\n".join(lines)
