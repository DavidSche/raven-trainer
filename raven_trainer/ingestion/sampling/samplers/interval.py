"""Interval Sampler — select frames at fixed intervals."""

from __future__ import annotations

from typing import Any

import cv2

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class IntervalSampler(BaseSampler):
    name = "interval"

    def __init__(self, interval: int = 30):
        self.interval = interval

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        meta.selected = (meta.frame_idx % self.interval == 0)
        return meta
