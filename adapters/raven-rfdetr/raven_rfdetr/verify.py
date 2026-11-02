"""RF-DETR pt-vs-onnx consistency verification (self-developed, P0-2).

The platform's ``onnx_verify.py`` imports ``ultralytics`` internally and is
NOT reusable for rfdetr ONNX — this module implements the same idea with the
rfdetr stack only:

1. Run the PyTorch model and the ONNX model (onnxruntime) on the same val
   images, compare top-k box IoU / score / class.
2. Return a ``{status, pt_vs_onnx}`` dict matching the builtin adapter's
   ``verification`` shape so ``cli/train.py``'s post-training verify chain
   works unchanged (verify failure is non-fatal, only recorded).

Requires ``onnxruntime`` (train extra pulls it via rfdetr deps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert normalized cxcywh boxes to normalized xyxy (in-place safe copy).

    rf-detr's ``pred_boxes`` are (center_x, center_y, width, height)
    normalized to [0,1] — see rfdetr.models.lwdetr.LWDETR.forward docstring.
    """
    b = np.asarray(boxes, dtype=np.float64).copy()
    if b.ndim == 1:
        b = b[None, :]
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2.0  # x1
    out[:, 1] = b[:, 1] - b[:, 3] / 2.0  # y1
    out[:, 2] = b[:, 0] + b[:, 2] / 2.0  # x2
    out[:, 3] = b[:, 1] + b[:, 3] / 2.0  # y2
    return out[0] if boxes.ndim == 1 else out


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU between two boxes in xyxy format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_val_images(dataset_yaml: str | Path, sample_count: int) -> List[Path]:
    """Locate up to sample_count val images for pt-vs-onnx comparison.

    Reuses ``ensure_rf_yolo_layout``'s projection (data root = the directory
    that holds ``data.yaml``) so verify agrees with train/evaluate on where
    the data lives — the raw Raven ``dataset.yaml`` ``path`` field is a
    project-relative logical path, not a reliable data root.
    """
    from .config import ensure_rf_yolo_layout

    yaml_path = Path(dataset_yaml)
    if not yaml_path.exists():
        return []

    try:
        projected = ensure_rf_yolo_layout(yaml_path.parent, yaml_path)
    except Exception:
        return []
    base = Path(projected["path"])
    val_rel = str(projected.get("val", "val/images"))
    val_dir = base / val_rel
    if not val_rel.endswith("images"):
        val_dir = val_dir / "images"
    if not val_dir.exists():
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(f for f in val_dir.iterdir() if f.suffix.lower() in exts)[:sample_count]


def verify_pt_vs_onnx(
    pt_path: str,
    onnx_path: str,
    dataset_yaml: Optional[str] = None,
    resolution: int = 640,
    iou_threshold: float = 0.85,
    score_tolerance: float = 0.05,
    sample_count: int = 10,
    top_k: int = 50,
    log_func: Optional[Callable[[str], None]] = None,
    pt_model: Any = None,
) -> Dict[str, Any]:
    """Verify ONNX export consistency against the PyTorch model (rfdetr-native).

    ``pt_path`` must be a full training checkpoint.  For weights-only state
    dicts (``{"model": ...}``), pass a pre-built RFDETR facade via
    ``pt_model`` instead (``pt_path`` is then unused).

    Returns dict like the builtin verification: ``{status, pt_vs_onnx, ...}``.
    ``status`` is ``"passed"``/``"failed"``/``"skipped"``/``"error"``.
    """
    import torch
    import yaml
    import onnxruntime as ort

    def _log(msg: str):
        if log_func:
            log_func(msg)

    result: Dict[str, Any] = {"status": "passed"}

    # ---- load models ----
    _log("[Verify] Loading PyTorch model...")
    if pt_model is not None:
        pt_wrapper = pt_model
    else:
        from rfdetr.detr import RFDETR

        pt_wrapper = RFDETR.from_checkpoint(pt_path)
    pt_nn = pt_wrapper.model.model if hasattr(pt_wrapper, "model") else pt_wrapper.model
    pt_nn.eval()

    _log("[Verify] Loading ONNX model (onnxruntime)...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    images = _find_val_images(dataset_yaml, sample_count) if dataset_yaml else []
    if not images:
        result["status"] = "skipped"
        result["error"] = "No val images found for pt-vs-onnx comparison"
        return result

    input_name = sess.get_inputs()[0].name
    max_iou_diff = 0.0
    max_score_diff = 0.0
    max_class_diff = 0
    compared = 0
    total_detections = 0
    mismatched_counts = 0

    for img_file in images:
        from PIL import Image

        img = Image.open(str(img_file)).convert("RGB")
        # 方形 resize（rf-detr 约定，非 letterbox）到 resolution
        img_resized = img.resize((resolution, resolution), Image.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        x = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)

        with torch.no_grad():
            out_pt = pt_nn(torch.from_numpy(x))
        # 注意：若该模型此前被切到 export 模式（export_rfdetr_onnx 的
        # LWDETR.export() 原地把 forward 换成 forward_export），输出是
        # (boxes, logits[, masks|keypoints]) 元组而非 dict；两者都兼容。
        if isinstance(out_pt, dict):
            boxes_pt = out_pt["pred_boxes"][0].cpu().numpy()
            logits_pt = out_pt["pred_logits"][0].cpu().numpy()
        else:
            boxes_pt = out_pt[0][0].cpu().numpy()
            logits_pt = out_pt[1][0].cpu().numpy()
        # pred_boxes 为归一化 cxcywh，需转 xyxy 再比较
        boxes_pt = _cxcywh_to_xyxy(boxes_pt)  # [Q, 4] xyxy norm
        scores_pt = logits_pt[:, :-1].max(axis=1)
        classes_pt = logits_pt[:, :-1].argmax(axis=1)

        out_on = sess.run(None, {input_name: x})
        boxes_on = _cxcywh_to_xyxy(np.asarray(out_on[0])[0])
        logits_on = np.asarray(out_on[1])[0]
        scores_on = logits_on[:, :-1].max(axis=1)
        classes_on = logits_on[:, :-1].argmax(axis=1)

        # top-k 对齐（按分数降序）
        def _topk(scores, boxes, classes, k):
            idx = np.argsort(-scores)[:k]
            return [(float(scores[i]), boxes[i].tolist(), int(classes[i])) for i in idx]

        top_pt = _topk(scores_pt, boxes_pt, classes_pt, top_k)
        top_on = _topk(scores_on, boxes_on, classes_on, top_k)
        total_detections += max(len(top_pt), len(top_on))
        if len(top_pt) != len(top_on):
            mismatched_counts += 1

        for (s_pt, b_pt, c_pt), (s_on, b_on, c_on) in zip(top_pt, top_on):
            iou = _compute_iou(b_pt, b_on)
            max_iou_diff = max(max_iou_diff, 1.0 - iou)
            max_score_diff = max(max_score_diff, abs(s_pt - s_on))
            max_class_diff = max(max_class_diff, abs(c_pt - c_on))
        compared += 1

    if compared == 0:
        result["status"] = "failed"
        result["pt_vs_onnx"] = {
            "error": "0 images compared — verification cannot confirm consistency",
            "images_compared": 0,
        }
        return result

    passed = (
        max_iou_diff <= (1.0 - iou_threshold)
        and max_score_diff <= score_tolerance
        and max_class_diff == 0
    )
    result["pt_vs_onnx"] = {
        "images_compared": compared,
        "total_detections_compared": total_detections,
        "mismatched_image_counts": mismatched_counts,
        "max_iou_diff": round(max_iou_diff, 6),
        "max_score_diff": round(max_score_diff, 6),
        "max_class_diff": int(max_class_diff),
        "passed": bool(passed),
    }
    result["status"] = "passed" if passed else "failed"
    _log(f"[Verify] pt_vs_onnx {'PASSED' if passed else 'FAILED'}: "
         f"max_iou_diff={max_iou_diff:.6f}, max_score_diff={max_score_diff:.6f}")
    return result
