"""Base Sampler interface for streaming pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2

from ..core import StreamContext, FrameMeta


class BaseSampler(ABC):
    """Base class for streaming samplers.

    Samplers process the video as a sequential stream:
      on_start()   — initialize state before the stream begins
      on_frame()   — called for each frame (sequential order)
      on_finish()  — finalize decisions after the stream ends
    """

    name: str = "base"

    def on_start(self, ctx: StreamContext, cap: cv2.VideoCapture) -> None:
        """Called once before the frame stream begins."""
        pass

    @abstractmethod
    def on_frame(self, ctx: StreamContext, frame: Any, meta: FrameMeta) -> FrameMeta:
        """Called for each frame. Returns (possibly modified) FrameMeta."""
        ...

    def on_finish(self, ctx: StreamContext) -> StreamContext:
        """Called after all frames have been streamed. Can modify decisions."""
        return ctx
