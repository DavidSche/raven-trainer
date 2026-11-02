"""RF-DETR evaluation + metric normalization.

Upstream ``RFDETR.evaluate()`` returns keys with a split prefix
(``val/mAP_50_95``, ``test/mAP_50``, ...) and ``mAR`` has NO ``_50_95``
suffix (verified against rfdetr 1.10.0.dev coco_eval callback).  We strip the
prefix and map to Raven conventions so ``cli/eval.py`` never sees an empty
dict (empty metrics => FAILED).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


#: rf-detr raw key (after prefix strip) -> Raven normalized key
_METRIC_KEY_MAP = {
    "mAP_50_95": "mAP50-95",
    "mAP_50": "mAP50",
    "mAP_75": "mAP75",
    "mAR": "mAR",            # 无 _50_95 后缀
    "F1": "f1",
    "precision": "precision",
    "recall": "recall",
    "segm_mAP_50_95": "segm_mAP50-95",
    "segm_mAP_50": "segm_mAP50",
    "segm_mAP_75": "segm_mAP75",
}


def normalize_metrics(raw: Dict[str, Any], split: str = "val") -> Dict[str, Any]:
    """Normalize rf-detr split-prefixed metric keys to Raven conventions.

    Strategy (P1-5): strip ``{split}/`` prefix, apply explicit mapping table,
    and route per-class ``AP/<name>`` to ``per_class/<name>``.  Unmatched keys
    go to ``raw_*`` so nothing is silently lost.
    """
    out: Dict[str, Any] = {}
    prefix = f"{split}/"
    for key, value in raw.items():
        if not isinstance(key, str):
            out[f"raw_{key}"] = value
            continue
        if key.startswith(prefix):
            stripped = key[len(prefix):]
        else:
            stripped = key  # 无前缀（防御）按原样处理

        if stripped.startswith("AP/"):
            out[f"per_class/{stripped[3:]}"] = value
        elif stripped.startswith("ema_"):
            out[f"raw_{key}"] = value  # ema_* 保留原样
        elif stripped in _METRIC_KEY_MAP:
            out[_METRIC_KEY_MAP[stripped]] = value
        else:
            out[f"raw_{key}"] = value

    # 清洗 numpy/torch 标量为 float，保证 JSON 序列化
    cleaned: Dict[str, Any] = {}
    for k, v in out.items():
        try:
            if hasattr(v, "item"):
                v = v.item()
            if hasattr(v, "tolist"):
                v = v.tolist()
            cleaned[k] = v
        except Exception:
            cleaned[k] = v
    return cleaned


def evaluate_rfdetr(
    dataset_cfg: str,
    weights: str,
    split: str = "val",
    device: str = "cpu",
    log_func: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate an RF-DETR checkpoint on a dataset split.

    Returns the RAW rf-detr metrics dict (split-prefixed); callers normalize
    via :func:`normalize_metrics`.
    """
    from rfdetr.detr import RFDETR

    from .train import _normalize_device

    _log = log_func or (lambda msg: None)
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # 数据集根取投影结果中的 path（data.yaml 所在目录）
    from .config import ensure_rf_yolo_layout
    projected = ensure_rf_yolo_layout(Path(dataset_cfg).parent, dataset_cfg, log_func=_log)
    dataset_dir = Path(projected["path"]).resolve()

    _log(f"[RFDETR] Loading checkpoint: {weights}")
    model = RFDETR.from_checkpoint(str(weights_path))
    _log(f"[RFDETR] Evaluating split='{split}' device='{device}'...")

    eval_kwargs: Dict[str, Any] = {
        "split": split,
        "dataset_dir": str(dataset_dir),
        "dataset_file": "yolo",
        "device": _normalize_device(device),
    }
    # 允许透传 batch_size / num_workers / output_dir 等（与 train 同参数面）
    for k in ("batch_size", "num_workers", "output_dir", "resolution"):
        if k in kwargs and kwargs[k] is not None:
            eval_kwargs[k] = kwargs[k]

    raw = model.evaluate(**eval_kwargs)
    if not raw:
        _log("[RFDETR] evaluate() returned no metrics")
        return {}
    _log(f"[RFDETR] raw metrics: {len(raw)} keys")
    return dict(raw)
