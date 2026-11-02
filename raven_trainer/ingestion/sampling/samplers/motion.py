"""Motion Sampler — MOG2 background subtraction, sequential streaming."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class MotionSampler(BaseSampler):
    name = "motion"

    def __init__(self, motion_area_threshold: float = 0.5):
        self.motion_area_threshold = motion_area_threshold
        self._mog2: Any = None

    def on_start(self, ctx: StreamContext, cap: cv2.VideoCapture) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2()

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        if self._mog2 is None:
            return meta
        fg_mask = self._mog2.apply(frame)
        motion_pct = float(np.count_nonzero(fg_mask) / fg_mask.size * 100)
        meta.motion_score = motion_pct
        if meta.selected:
            meta.selected = motion_pct >= self.motion_area_threshold
        return meta
