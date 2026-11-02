"""Object-Driven Sampler — YOLO-based (requires raven-ultralytics)."""

from __future__ import annotations

from typing import Any, List, Optional

import cv2

from ..core import StreamContext, FrameMeta
from .base import BaseSampler


class ObjectDrivenSampler(BaseSampler):
    name = "object"

    def __init__(
        self,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.4,
        background_interval_sec: int = 10,
    ):
        self.target_classes = target_classes or ["person"]
        self.confidence_threshold = confidence_threshold
        self.background_interval_sec = background_interval_sec
        self._detector = None
        self._frames_since_obj = 0

    def _get_detector(self):
        if self._detector is None:
            try:
                from ...models import FrameDetector
                self._detector = FrameDetector(conf_threshold=self.confidence_threshold)
            except ImportError:
                self._detector = None
        return self._detector

    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        detector = self._get_detector()
        if detector is None:
            return meta

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

        try:
            dets = detector.detect(tmp_path)
            has_target = any(d["label"] in self.target_classes for d in dets)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if has_target:
            self._frames_since_obj = 0
            meta.object_score = 1.0
        else:
            self._frames_since_obj += 1
            meta.object_score = 0.0
            bg_interval_frames = int(ctx.fps * self.background_interval_sec)
            if self._frames_since_obj < bg_interval_frames and meta.selected:
                meta.selected = False

        return meta
