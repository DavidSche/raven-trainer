"""Lightweight YOLO26 detector for event-based sampling.

When a person enters ROI, start saving frames at high FPS.
When no person detected, fall back to low FPS (1 frame/min).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class FrameDetector:
    """Wrapper around YOLO26 for frame-level object detection.

    Used by event-based sampling to detect persons entering ROI.
    Detections are cached to avoid re-running on identical frames.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        self.model_path = model_path or "yolo26n.pt"
        self.conf_threshold = conf_threshold
        self._model: Any = None

    def _lazy_load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def detect(self, image_path: str | Path) -> List[Dict[str, Any]]:
        """Run detection on a single image. Returns list of detections.

        Each detection:
          {"label": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
        """
        self._lazy_load()
        results = self._model(str(image_path), conf=self.conf_threshold, verbose=False)
        dets: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                dets.append({
                    "label": r.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [float(x) for x in box],
                })
        return dets

    def has_person(self, image_path: str | Path) -> bool:
        """Quick check: is there a person in this frame?"""
        dets = self.detect(image_path)
        return any(d["label"] == "person" for d in dets)
