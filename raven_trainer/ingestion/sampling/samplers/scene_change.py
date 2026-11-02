"""Scene Change Sampler — histogram-based scene change detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class SceneChangeSampler(BaseSampler):
    name = "scene_change"

    def __init__(self, method: str = "histogram", threshold: float = 0.3):
        self.method = method
        self.threshold = threshold
        self._prev_hist = None
        self._first = True

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        score = 0.0
        if self._prev_hist is not None:
            score = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CHISQR))
            if meta.selected:
                meta.selected = score >= self.threshold
        self._prev_hist = hist

        meta.scene_score = score
        return meta

    def on_finish(self, ctx: StreamContext) -> StreamContext:
        # Always keep first frame
        if ctx.frames:
            ctx.frames[0].selected = True
        return ctx
