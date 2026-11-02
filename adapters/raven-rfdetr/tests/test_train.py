"""Unit tests for raven_rfdetr.train — checkpoint location + metrics.csv tail.

No rfdetr install needed: the module-level rfdetr import happens lazily inside
``train_rfdetr``, so we can test the pure helpers directly.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven_rfdetr.train import (  # noqa: E402
    _locate_best_checkpoint,
    _locate_last_checkpoint,
    _normalize_device,
    _read_last_metrics_row,
    _tail_metrics_csv,
)


# ---------------------------------------------------------------------------
# device normalization (上游 cuda:N -> list devices bug 规避)
# ---------------------------------------------------------------------------

def test_normalize_device_cuda_index_stripped():
    assert _normalize_device("cuda:0") == "cuda"
    assert _normalize_device("cuda:1") == "cuda"
    assert _normalize_device("mps:0") == "mps"


def test_normalize_device_passthrough():
    assert _normalize_device("cpu") == "cpu"
    assert _normalize_device("cuda") == "cuda"
    assert _normalize_device(None) is None


# ---------------------------------------------------------------------------
# checkpoint location (P1-6): .pth 交付 vs .ckpt 续训
# ---------------------------------------------------------------------------

def test_locate_best_checkpoint_pth(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint_best_total.pth").write_bytes(b"x")
    assert _locate_best_checkpoint(run_dir) == run_dir / "checkpoint_best_total.pth"


def test_locate_best_ema_fallback(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint_best_ema.pth").write_bytes(b"x")
    assert _locate_best_checkpoint(run_dir) == run_dir / "checkpoint_best_ema.pth"


def test_locate_best_returns_none_when_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _locate_best_checkpoint(run_dir) is None


def test_locate_last_checkpoint_ckpt(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.ckpt").write_bytes(b"x")
    assert _locate_last_checkpoint(run_dir) == run_dir / "last.ckpt"


def test_locate_last_checkpoint_ema_fallback(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last_ema.pth").write_bytes(b"x")
    assert _locate_last_checkpoint(run_dir) == run_dir / "last_ema.pth"


# ---------------------------------------------------------------------------
# metrics.csv tail (P1-7)
# ---------------------------------------------------------------------------

def _write_metrics_csv(path: Path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_read_last_metrics_row(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path, [
        {"epoch": "0", "train/loss": "2.1", "val/mAP_50_95": "0.1"},
        {"epoch": "1", "train/loss": "1.8", "val/mAP_50_95": "0.2"},
    ])
    row = _read_last_metrics_row(csv_path)
    assert row["epoch"] == 1.0
    assert row["val/mAP_50_95"] == 0.2


def test_read_last_metrics_row_missing_file(tmp_path):
    assert _read_last_metrics_row(tmp_path / "nope.csv") == {}


def test_tail_metrics_csv_pushes_new_epochs(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path, [
        {"epoch": "0", "train/loss": "2.1", "val/mAP_50_95": "0.1"},
        {"epoch": "1", "train/loss": "1.8", "val/mAP_50_95": "0.2"},
    ])

    received = []
    stop = threading.Event()
    t = threading.Thread(
        target=_tail_metrics_csv,
        args=(csv_path, received.append, stop),
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2)

    epochs = {r.get("epoch") for r in received}
    assert 1 in epochs  # 至少收到 epoch 1
    # 每行都含归一化键
    if received:
        last = received[-1]
        assert "train/loss" in last
