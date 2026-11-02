"""RF-DETR training orchestration.

- wraps ``RFDETR<Size>().train(...)`` (returns None) into a CLI-consumable dict
- reports progress via ``progress_bar="tqdm"`` + a background thread tailing
  ``output_dir/metrics.csv`` (CSVLogger always writes it) -> ``metric_func``
- locates checkpoint artifacts (best .pth / last .ckpt) after training
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import VariantInfo, filter_train_kwargs


def _normalize_device(device: Any) -> str:
    """Work around rf-detr 1.10.0.dev bug (detr.py:787-820 + trainer.py:193-207):
    ``device="cuda:N"`` maps to PTL ``devices=[N]`` (a list), but
    ``build_trainer._requests_multiple_devices`` only handles int/str and
    crashes on list (``'list' object has no attribute 'strip'``).

    Normalize ``cuda:N`` / ``mps:N`` to the bare accelerator name so upstream
    receives ``devices=None`` (single-device PTL auto-selection).  Multi-GPU
    index selection is deferred to the PTL ``build_trainer`` escape hatch (see
    plan §2.3 — not used in v1).
    """
    if isinstance(device, str) and ":" in device:
        head, _, _ = device.partition(":")
        if head in ("cuda", "mps"):
            return head
    return device


#: columns we care about from metrics.csv (verified against rf-detr CSVLogger)
_METRIC_COLUMNS = (
    "epoch", "train/loss", "val/loss",
    "val/mAP_50_95", "val/mAP_50", "val/mAP_75", "val/mAR",
    "val/F1", "val/precision", "val/recall",
)


def _tail_metrics_csv(
    metrics_csv: Path,
    metric_func: Optional[Callable[[Dict[str, Any]], None]],
    stop_event: threading.Event,
    poll_interval: float = 2.0,
) -> None:
    """Background thread: tail metrics.csv and push per-epoch metrics.

    metrics.csv is rewritten by CSVLogger each epoch, so we track the largest
    epoch already reported and only emit new rows.
    """
    if metric_func is None:
        return
    last_epoch = -1
    while not stop_event.is_set():
        try:
            if not metrics_csv.exists():
                time.sleep(poll_interval)
                continue
            with open(metrics_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                try:
                    epoch = int(float(row.get("epoch", "-1")))
                except (TypeError, ValueError):
                    continue
                if epoch <= last_epoch:
                    continue
                last_epoch = epoch
                metric_data: Dict[str, Any] = {"epoch": epoch}
                for col in _METRIC_COLUMNS:
                    raw = row.get(col, "")
                    if raw not in (None, ""):
                        try:
                            metric_data[col] = float(raw)
                        except (TypeError, ValueError):
                            metric_data[col] = raw
                try:
                    metric_func(metric_data)
                except Exception:
                    pass  # 指标推送失败不影响训练
        except Exception:
            pass
        time.sleep(poll_interval)


def _locate_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Locate the deliverable checkpoint (best .pth; .ckpt is resume-only)."""
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "last_ema.pth",
    ):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _locate_last_checkpoint(run_dir: Path) -> Optional[Path]:
    for name in ("last.ckpt", "last_ema.pth"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def train_rfdetr(
    dataset_cfg: str,
    variant: VariantInfo,
    config: Dict[str, Any],
    log_func: Optional[Callable[[str], None]] = None,
    metric_func: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run RF-DETR training. Returns ``{model_path, run_dir, metrics}``."""
    from rfdetr import variants as _rf_variants

    _log = log_func or (lambda msg: None)

    # data.yaml 已由调用方生成；数据集根取投影结果中的 path（data.yaml 所在目录）
    from .config import ensure_rf_yolo_layout
    projected = ensure_rf_yolo_layout(Path(dataset_cfg).parent, dataset_cfg, log_func=_log)
    dataset_dir = Path(projected["path"]).resolve()
    _log(f"[RFDETR] Training variant: {variant.variant_cls} "
         f"(resolution={variant.resolution}, seg={variant.is_seg}, "
         f"deprecated={variant.deprecated})")

    if variant.deprecated:
        _log("[RFDETR] WARNING: RFDETRBase is deprecated (removed in 2.0), "
             "not recommended for new projects")

    output_dir = Path(config.pop("output_dir", dataset_dir / "runs" / "rfdetr"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # resolution: default from variant, overridable via train.yaml (imgsz/resolution)
    resolution = int(config.pop("resolution", config.pop("imgsz", variant.resolution)))
    device = _normalize_device(config.pop("device", "cpu"))

    # num_classes from data nc (Raven 数据血缘), fallback to auto-detect
    num_classes = config.pop("num_classes", config.pop("nc", None))
    pretrain_weights = config.pop("pretrain_weights", None)
    epochs = int(config.pop("epochs", 100))

    # progress_bar: 默认 None 时 PTL 关闭进度条；显式 tqdm 才能拿到进度行
    if "progress_bar" not in config:
        config["progress_bar"] = "tqdm"

    # Filter against TrainConfig whitelist (extra="forbid")
    train_kwargs = filter_train_kwargs(config, log_func=_log)
    train_kwargs.update({
        "dataset_dir": str(dataset_dir),
        "dataset_file": "yolo",
        "output_dir": str(output_dir),
        "epochs": epochs,
    })

    _log(f"[RFDETR] model_class={variant.variant_cls}, resolution={resolution}, "
         f"device={device}, num_classes={num_classes}, epochs={epochs}")
    _log(f"[RFDETR] output_dir={output_dir}")

    # construct variant class
    model_cls = getattr(_rf_variants, variant.variant_cls)
    init_kwargs: Dict[str, Any] = {}
    if pretrain_weights:
        init_kwargs["pretrain_weights"] = pretrain_weights
    if num_classes is not None:
        init_kwargs["num_classes"] = int(num_classes)
    _log(f"[RFDETR] init kwargs: {init_kwargs}")
    model = model_cls(**init_kwargs)

    # tail metrics.csv in background while training
    metrics_csv = output_dir / "metrics.csv"
    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_metrics_csv,
        args=(metrics_csv, metric_func, stop_event),
        daemon=True,
        name="rfdetr-metrics-tail",
    )
    tail_thread.start()
    try:
        model.train(
            device=device,
            resolution=resolution,
            **train_kwargs,
        )
    finally:
        stop_event.set()
        tail_thread.join(timeout=5)

    _log("[RFDETR] Training completed.")

    # training metrics from final metrics.csv row (CSVLogger 恒写)
    metrics: Dict[str, Any] = _read_last_metrics_row(metrics_csv)
    _log(f"[RFDETR] final metrics: {metrics}")

    best = _locate_best_checkpoint(output_dir)
    last = _locate_last_checkpoint(output_dir)

    result: Dict[str, Any] = {
        "model_path": str(best) if best else None,
        "run_dir": str(output_dir),
        "metrics": metrics,
    }
    if best:
        _log(f"[RFDETR] best checkpoint: {best}")
    if last:
        result["last_checkpoint"] = str(last)
        _log(f"[RFDETR] last checkpoint (resume): {last}")
    if metrics_csv.exists():
        result["metrics_csv"] = str(metrics_csv)
    return result


def _read_last_metrics_row(metrics_csv: Path) -> Dict[str, Any]:
    """Read the last metrics.csv row as a flat dict (defensive fallback)."""
    if not metrics_csv.exists():
        return {}
    try:
        with open(metrics_csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        row = rows[-1]
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if k and v not in (None, ""):
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
        return out
    except Exception:
        return {}
