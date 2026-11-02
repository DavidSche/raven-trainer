"""Sampling Pipeline -- uses ffmpeg for frame extraction (faster + HEVC reliable),
falls back to OpenCV seeking when ffmpeg is not available.

DecodeConfig controls HOW frame positions are determined:
  - interval:  evenly-spaced positions (every_n_frames gap)
  - fps:       positions at target frames-per-second
  - keyframe:  positions at I-frame boundaries (via ffprobe)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time as _stime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .core import StreamContext, FrameMeta, DecodeConfig
from .samplers.base import BaseSampler
from .samplers.interval import IntervalSampler
from .samplers.motion import MotionSampler
from .samplers.scene_change import SceneChangeSampler
from .samplers.object_driven import ObjectDrivenSampler
from .samplers.burst import BurstSampler
from .samplers.balance import BalanceSampler


SAMPLER_REGISTRY = {
    "interval": IntervalSampler,
    "motion": MotionSampler,
    "scene_change": SceneChangeSampler,
    "object": ObjectDrivenSampler,
    "burst": BurstSampler,
    "balance": BalanceSampler,
}


def build_pipeline(stages: list[str]) -> list[BaseSampler]:
    pipeline = []
    for name in stages:
        cls = SAMPLER_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLER_REGISTRY)}")
        pipeline.append(cls())
    return pipeline


def _ffmpeg_extract_frame(video_path: str, frame_idx: int, fps: float, output_path: str) -> bool:
    ts = frame_idx / fps
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        return os.path.getsize(output_path) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, subprocess.CalledProcessError):
        return False


def _has_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── Position computation based on DecodeConfig ──

def _compute_positions(
    total_frames: int,
    fps: float,
    sample_count: int,
    decode_config: Optional[DecodeConfig] = None,
    video_path: str = "",
) -> list[int]:
    """Compute frame positions to visit based on decode strategy.

    M-40: ``video_path`` is now passed through to the keyframe branch so
    ``_compute_keyframe_positions`` can actually invoke ``ffprobe`` on the
    real file. Previously it received an empty string and silently fell
    back to the interval-mode heuristic, making the ``keyframe`` decode
    mode a no-op.
    """
    if decode_config is None:
        decode_config = DecodeConfig()

    if decode_config.mode == "keyframe":
        return _compute_keyframe_positions(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            sample_count=sample_count,
        )

    if decode_config.mode == "fps" and decode_config.target_fps > 0:
        target_fps = decode_config.target_fps
        step = max(1, int(fps / target_fps))
        positions = list(range(0, total_frames, step))
        # Clamp to sample_count
        if len(positions) > sample_count:
            sub_step = max(1, len(positions) // sample_count)
            positions = positions[::sub_step][:sample_count]
        return positions

    # Default: interval mode (legacy-compatible)
    every_n = decode_config.every_n_frames
    positions = list(range(0, total_frames, max(1, every_n)))
    # Clamp to sample_count
    if sample_count > 0 and len(positions) > sample_count:
        sub_step = max(1, len(positions) // sample_count)
        positions = positions[::sub_step][:sample_count]
    return positions


def _compute_keyframe_positions(
    video_path: str,
    fps: float,
    total_frames: int,
    sample_count: int,
) -> list[int]:
    """Use ffprobe to locate keyframes (I-frames)."""
    if not video_path or not os.path.exists(video_path):
        # Fallback: interval mode
        return list(range(0, total_frames, max(1, total_frames // max(sample_count, 1))))[:sample_count]

    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-skip_frame", "nokey",
            "-show_frames", "-show_entries", "frame=coded_picture_number",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        frames_info = data.get("frames", [])
        key_positions = [int(f["coded_picture_number"]) for f in frames_info]

        if not key_positions:
            # Fallback
            return list(range(0, total_frames, max(1, total_frames // max(sample_count, 1))))[:sample_count]

        # Take sample_count evenly from keyframe positions
        if len(key_positions) > sample_count:
            step = max(1, len(key_positions) // sample_count)
            key_positions = key_positions[::step][:sample_count]

        return sorted(set(key_positions))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, IndexError):
        # Fallback: interval mode
        return list(range(0, total_frames, max(1, total_frames // max(sample_count, 1))))[:sample_count]


# ── Main pipeline runner ──

def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    pipeline: list[BaseSampler],
    fps: float = 25.0,
    total_frames: int = 0,
    sample_count: int = 100,
    decode_config: Optional[DecodeConfig] = None,
    alt_positions: Optional[list[int]] = None,  # explicit override, e.g. from caller
) -> List[FrameMeta]:
    """Run the sampling pipeline over video frames.

    Parameters
    ----------
    decode_config : DecodeConfig, optional
        Controls frame-position computation (interval / fps / keyframe).
        If None, falls back to legacy interval mode.
    alt_positions : list[int], optional
        If provided, bypasses decode_config and uses these positions directly.
    """
    use_ffmpeg = _has_ffmpeg()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # M-18: Wrap the entire decoding loop in try/finally so cap.release() runs
    # even if a sampler raises, ffmpeg crashes, or the user interrupts the run.
    # Previously release() only ran on the happy path, leaking the video handle
    # and keeping the file locked on Windows.
    try:
        if fps <= 0 or total_frames <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # If decode_config not provided, use defaults
        if decode_config is None:
            decode_config = DecodeConfig()

        ctx = StreamContext(
            video_path=Path(video_path),
            fps=fps,
            total_frames=total_frames,
            output_dir=out,
            decode_config=decode_config,
        )

        source_name = ctx.source_name
        frame_dir = out / source_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Determine frame positions
        if alt_positions is not None:
            positions = alt_positions
        else:
            sample_count = min(sample_count, total_frames)
            # M-40: pass the real video path so the keyframe decode mode can
            # actually run ffprobe instead of falling back to the interval
            # heuristic with an empty path.
            positions = _compute_positions(
                total_frames, fps, sample_count, decode_config, str(video_path)
            )

        jpeg_quality = decode_config.jpeg_quality

        t0 = _stime.time()
        last_pct = 0

        for sampler in pipeline:
            sampler.on_start(ctx, cap)

        method = "ffmpeg" if use_ffmpeg else "opencv"
        decode_mode = decode_config.mode
        print(f"    [{_stime.strftime('%H:%M:%S')}] Sampling {len(positions)} positions from {total_frames} frames ({method}, decode={decode_mode})")

        all_meta: List[FrameMeta] = []
        saved = 0

        for idx, pos in enumerate(positions):
            if pos >= total_frames:
                continue

            frame = None
            ret = False

            # Use ffmpeg for frame extraction (reliable HEVC decoding)
            if use_ffmpeg:
                # M-38: Use a unique temp file per call. The previous fixed
                # name "raven_frame_{pos}.jpg" collided across concurrent
                # run_pipeline invocations (e.g. multiple videos extracted
                # in parallel), causing one run to clobber another's frame
                # or to read a half-written file.
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_file = tmp.name
                if _ffmpeg_extract_frame(str(video_path), pos, fps, tmp_file):
                    frame = cv2.imread(tmp_file)
                    ret = frame is not None
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass

            # Fallback to OpenCV seeking
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()

            if not ret or frame is None:
                continue

            meta = FrameMeta(
                source_name=source_name,
                frame_idx=pos,
                timestamp_sec=pos / fps if fps > 0 else 0,
                selected=True,
            )

            for sampler in pipeline:
                meta = sampler.on_frame(ctx, frame, meta)

            all_meta.append(meta)

            if meta.selected:
                fp = frame_dir / f"{saved:06d}.jpg"
                cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                meta.saved_path = fp
                saved += 1

            pct = (idx + 1) * 100 // len(positions)
            if pct - last_pct >= 10:
                elapsed = _stime.time() - t0
                rate = (idx + 1) / max(elapsed, 0.01)
                eta = (len(positions) - idx - 1) / max(rate, 1)
                print(f"      [{_stime.strftime('%H:%M:%S')}] {pct}% ({idx+1}/{len(positions)})  {rate:.0f} fps  ETA: {eta:.0f}s  saved: {saved}")
                last_pct = pct

        ctx.frames = all_meta
        for sampler in pipeline:
            ctx = sampler.on_finish(ctx)
    finally:
        cap.release()

    elapsed = _stime.time() - t0
    rate = len(positions) / max(elapsed, 0.01)
    print(f"    [{_stime.strftime('%H:%M:%S')}] Done: {saved} frames saved in {elapsed:.1f}s ({rate:.0f} samples/s)")
    print(f"    Output: {frame_dir}")
    return [m for m in all_meta if m.selected and m.saved_path is not None]


__all__ = [
    "BaseSampler", "build_pipeline", "run_pipeline",
    "SAMPLER_REGISTRY",
    "IntervalSampler", "MotionSampler", "SceneChangeSampler",
    "ObjectDrivenSampler", "BurstSampler", "BalanceSampler",
]
