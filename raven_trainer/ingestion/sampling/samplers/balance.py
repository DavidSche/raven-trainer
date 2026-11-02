"""Balance Sampler — limit max frames."""

from __future__ import annotations

from typing import Any

import cv2

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class BalanceSampler(BaseSampler):
    name = "balance"

    def __init__(self, max_per_camera: int = 5000):
        self.max_per_camera = max_per_camera

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        return meta

    def on_finish(self, ctx: StreamContext) -> StreamContext:
        frames = ctx.frames
        if not frames:
            return ctx

        selected = [m for m in frames if m.selected]
        if len(selected) <= self.max_per_camera:
            return ctx

        # Evenly subsample
        step = len(selected) / self.max_per_camera
        keep_indices = set()
        for i in range(self.max_per_camera):
            idx = int(i * step)
            if idx < len(selected):
                keep_indices.add(selected[idx].frame_idx)

        for m in frames:
            if m.frame_idx not in keep_indices:
                m.selected = False

        print(f"      BalanceSampler: {len(selected)} -> {self.max_per_camera} frames")
        return ctx
