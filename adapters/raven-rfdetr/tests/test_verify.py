"""Unit tests for raven_rfdetr.verify helpers (no torch/onnxruntime needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven_rfdetr.verify import _compute_iou, _cxcywh_to_xyxy  # noqa: E402


def test_cxcywh_to_xyxy_single():
    # center (0.5, 0.5), w=0.4, h=0.2 -> x1=0.3, y1=0.4, x2=0.7, y2=0.6
    out = _cxcywh_to_xyxy(np.array([0.5, 0.5, 0.4, 0.2]))
    np.testing.assert_allclose(out, [0.3, 0.4, 0.7, 0.6])


def test_cxcywh_to_xyxy_batch():
    boxes = np.array([[0.5, 0.5, 0.4, 0.2], [0.25, 0.75, 0.1, 0.1]])
    out = _cxcywh_to_xyxy(boxes)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], [0.3, 0.4, 0.7, 0.6])
    np.testing.assert_allclose(out[1], [0.2, 0.7, 0.3, 0.8])


def test_compute_iou_identical():
    assert _compute_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_compute_iou_disjoint():
    assert _compute_iou([0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1]) == 0.0


def test_compute_iou_half_overlap():
    # [0,0,1,1] 与 [0.5,0,1.5,1] -> 交 0.5，并 1.5 -> 1/3
    iou = _compute_iou([0, 0, 1, 1], [0.5, 0, 1.5, 1])
    assert abs(iou - 1.0 / 3.0) < 1e-9
