"""Layer 1: Import -- build index of raw sources.

Does NOT extract frames. Only:
  1. Scans raw/ directory for videos and image folders
  2. Reads metadata (fps, resolution, duration, sha256)
  3. Writes metadata.json per source
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2

from ..core import RawSource, RawSourceMetadata


class RawImporter:
    """Scan raw/ directory and build metadata index."""

    def __init__(self, raw_root: str | Path):
        self.raw_root = Path(raw_root)

    def scan(self) -> List[RawSource]:
        sources: List[RawSource] = []
        if not self.raw_root.exists():
            return sources

        for entry in sorted(self.raw_root.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                meta = self._build_video_metadata(entry)
                sources.append(RawSource(path=entry, metadata=meta))
            elif entry.is_dir():
                if _is_image_dir(entry):
                    meta = self._build_image_dir_metadata(entry)
                    sources.append(RawSource(path=entry, metadata=meta))
        return sources

    def _build_video_metadata(self, path: Path) -> RawSourceMetadata:
        """Read video metadata using OpenCV."""
        meta = RawSourceMetadata(
            source_name=path.stem,
            source_type="video",
            path=str(path.relative_to(self.raw_root)),
            file_size_mb=round(path.stat().st_size / (1024 * 1024), 1),
        )

        # Try ffprobe first (more accurate)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        meta.fps = _parse_fps(stream.get("r_frame_rate", ""))
                        meta.resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
                        meta.total_frames = int(stream.get("nb_frames", 0)) or None
                        break
                duration = data.get("format", {}).get("duration")
                if duration:
                    meta.duration_sec = round(float(duration), 1)
                creation_time = data.get("format", {}).get("tags", {}).get("creation_time")
                if creation_time:
                    meta.recording_time = creation_time
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        # Fallback: use OpenCV
        if meta.fps is None or meta.total_frames is None:
            cap = cv2.VideoCapture(str(path))
            # M-17: ensure cap.release() runs even if any cv2 call raises
            # (e.g. on a truncated video). Previously release() only ran on the
            # happy path, leaking the underlying video handle.
            try:
                if cap.isOpened():
                    meta.fps = round(cap.get(cv2.CAP_PROP_FPS), 2) or None
                    meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if w and h:
                        meta.resolution = f"{w}x{h}"
                    if meta.fps and meta.total_frames:
                        meta.duration_sec = round(meta.total_frames / meta.fps, 1)
            finally:
                cap.release()

        # SHA256
        meta.sha256 = _compute_sha256(path)

        return meta

    def _build_image_dir_metadata(self, path: Path) -> RawSourceMetadata:
        images = sorted([f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        return RawSourceMetadata(
            source_name=path.name,
            source_type="image_dir",
            path=str(path.relative_to(self.raw_root)),
            total_frames=len(images),
        )

    def save_metadata(self, source: RawSource) -> None:
        if source.metadata is None:
            return
        meta_path = source.path.parent / f"{source.metadata.source_name}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(source.metadata.to_dict(), f, indent=2, ensure_ascii=False)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _is_image_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = list(path.iterdir())[:50]
    if not files:
        return False
    img_count = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTS)
    return img_count / len(files) > 0.5


def _parse_fps(r_frame_rate: str) -> Optional[float]:
    """Parse ffprobe frame rate string like '30000/1001' -> 29.97."""
    try:
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            return round(float(num) / float(den), 2)
        return float(r_frame_rate)
    except (ValueError, ZeroDivisionError):
        return None


def _compute_sha256(path: Path, sample_size: int = 65536) -> str:
    """Compute SHA256 of first 64KB (fast, good enough for dedup)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(sample_size)
        h.update(chunk)
    return h.hexdigest()
