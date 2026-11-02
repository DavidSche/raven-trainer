"""Unit tests for raven_rfdetr.evaluate.normalize_metrics (P1-5)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven_rfdetr.evaluate import normalize_metrics  # noqa: E402


def test_normalize_val_prefix():
    raw = {
        "val/mAP_50_95": 0.42,
        "val/mAP_50": 0.60,
        "val/mAP_75": 0.45,
        "val/mAR": 0.55,
        "val/F1": 0.58,
        "val/precision": 0.62,
        "val/recall": 0.54,
    }
    out = normalize_metrics(raw, split="val")
    assert out["mAP50-95"] == 0.42
    assert out["mAP50"] == 0.60
    assert out["mAP75"] == 0.45
    assert out["mAR"] == 0.55  # mAR 无 _50_95 后缀
    assert out["f1"] == 0.58
    assert out["precision"] == 0.62
    assert out["recall"] == 0.54


def test_normalize_test_prefix():
    raw = {"test/mAP_50_95": 0.31, "test/mAR": 0.40, "test/F1": 0.44}
    out = normalize_metrics(raw, split="test")
    assert out["mAP50-95"] == 0.31
    assert out["mAR"] == 0.40
    assert out["f1"] == 0.44


def test_normalize_per_class_ap():
    raw = {
        "val/AP/person": 0.71,
        "val/AP/helmet": 0.55,
    }
    out = normalize_metrics(raw, split="val")
    assert out["per_class/person"] == 0.71
    assert out["per_class/helmet"] == 0.55


def test_normalize_ema_keys_preserved_raw():
    raw = {"val/ema_mAP_50_95": 0.44, "val/mAP_50_95": 0.42}
    out = normalize_metrics(raw, split="val")
    assert out["raw_val/ema_mAP_50_95"] == 0.44  # ema_* 保留原样
    assert out["mAP50-95"] == 0.42


def test_normalize_unknown_keys_to_raw():
    raw = {"val/some_future_metric": 123}
    out = normalize_metrics(raw, split="val")
    assert out["raw_val/some_future_metric"] == 123


def test_normalize_segm_metrics():
    raw = {"val/segm_mAP_50_95": 0.38, "val/segm_mAP_50": 0.52}
    out = normalize_metrics(raw, split="val")
    assert out["segm_mAP50-95"] == 0.38
    assert out["segm_mAP50"] == 0.52


def test_normalize_tensor_scalar_coercion():
    """numpy/torch 标量清洗为 float，保证 JSON 可序列化。"""
    import numpy as np
    raw = {"val/mAP_50_95": np.float32(0.42)}
    out = normalize_metrics(raw, split="val")
    assert isinstance(out["mAP50-95"], float)
    assert abs(out["mAP50-95"] - 0.42) < 1e-5


def test_normalize_empty_returns_empty():
    assert normalize_metrics({}, split="val") == {}
