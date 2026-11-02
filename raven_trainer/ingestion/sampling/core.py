"""Core data models for the Sampling Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DecodeConfig:
    """Video decode strategy — determines HOW frames are read from the source.

    mode: interval | fps | keyframe
      - interval:  extract every N frames
      - fps:       extract at target frames-per-second
      - keyframe:  extract only keyframes (I-frames)
    """
    mode: str = "interval"
    every_n_frames: int = 30
    target_fps: float = 0.0          # 0.0 = use source fps
    jpeg_quality: int = 80            # CV_IMWRITE_JPEG_QUALITY (1-100)


@dataclass
class FrameMeta:
    """Metadata about a single frame in the stream."""
    frame_idx: int
    timestamp_sec: float = 0.0
    source_name: str = ""
    selected: bool = True
    motion_score: float = 0.0
    scene_score: float = 0.0
    object_score: float = 0.0
    saved_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamContext:
    """Context passed through the streaming pipeline."""
    video_path: Path
    output_dir: Path
    fps: float
    total_frames: int
    source_name: str = ""
    frames: List[FrameMeta] = field(default_factory=list)
    decode_config: Optional[DecodeConfig] = None

    def __post_init__(self):
        self.source_name = self.video_path.stem
