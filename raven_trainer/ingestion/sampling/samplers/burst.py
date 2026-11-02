"""Burst Sampler — expand selected frames with context windows."""

from __future__ import annotations

from typing import Any

import cv2

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class BurstSampler(BaseSampler):
    name = "burst"

    def __init__(self, before: int = 5, after: int = 10):
        self.before = before
        self.after = after

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        return meta

    def on_finish(self, ctx: StreamContext) -> StreamContext:
        frames = ctx.frames
        if not frames:
            return ctx

        # Build sorted list of all sampled frame indices
        all_idxs = sorted([m.frame_idx for m in frames])
        idx_set = set(all_idxs)
        selected_idxs = {m.frame_idx for m in frames if m.selected}

        if not selected_idxs:
            return ctx

        original_count = len(selected_idxs)
        expanded = set(selected_idxs)

        for idx in selected_idxs:
            position = all_idxs.index(idx) if idx in idx_set else -1
            if position < 0:
                continue
            # Expand to neighbors in the sparse sampled list
            for offset in range(-self.before, self.after + 1):
                neighbor_pos = position + offset
                if 0 <= neighbor_pos < len(all_idxs):
                    expanded.add(all_idxs[neighbor_pos])

        for m in frames:
            if m.frame_idx in expanded:
                m.selected = True

        print(f"      BurstSampler: {original_count} -> {len(expanded)} frames")
        return ctx
