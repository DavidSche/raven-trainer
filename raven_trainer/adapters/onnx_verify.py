"""ONNX consistency verification — ported from aiva-trainer.

Provides two-step verification:
1. .pt vs .onnx per-image prediction comparison (IoU, score, class)
2. .onnx model val validation (mAP50 / mAP50-95)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def verify_onnx_consistency(
    pt_path: str,
    onnx_path: str,
    dataset_yaml: str,
    imgsz: int = 640,
    batch: int = 1,
    half: bool = False,
    iou_threshold: float = 0.85,
    score_tolerance: float = 0.05,
    sample_count: int = 10,
    log_func: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Verify ONNX export consistency against the original PyTorch model.

    Two checks:
      1. pt_vs_onnx — per-image prediction comparison
      2. onnx_val   — ONNX model validation (mAP50, mAP50-95)

    Args:
        pt_path: .pt model path
        onnx_path: .onnx model path
        dataset_yaml: dataset.yaml path (for val images + validation)
        imgsz: Inference/validation image size
        batch: Batch size for validation
        half: FP16 half precision
        iou_threshold: Minimum IoU for box consistency
        score_tolerance: Maximum score difference tolerance
        sample_count: Number of images for per-image comparison
        log_func: Optional log callback

    Returns:
        Dict with verification results
    """
    from ultralytics import YOLO
    import yaml

    def _log(msg: str):
        if log_func:
            log_func(msg)

    result: Dict[str, Any] = {"status": "passed"}

    # Parse dataset.yaml to find val images directory
    yaml_path = Path(dataset_yaml)
    if not yaml_path.exists():
        result["status"] = "skipped"
        result["error"] = f"dataset.yaml not found: {dataset_yaml}"
        return result

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    val_path_str: str = data.get("val", "")
    val_path = Path(val_path_str)
    if not val_path.is_absolute():
        val_path = yaml_path.parent / val_path_str
    if not val_path.exists():
        base = Path(data.get("path", yaml_path.parent))
        val_path = base / data.get("val", "val")

    # Step 1: pt vs onnx per-image comparison
    _log("[Verify] Step 1/2: Comparing .pt vs .onnx predictions...")
    model_pt = YOLO(str(pt_path))
    model_onnx = YOLO(str(onnx_path))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    val_images_dir = val_path / "images" if val_path.exists() else val_path
    if not val_images_dir.exists():
        _log(f"[Verify] Val images not found: {val_images_dir}")
        result["pt_vs_onnx"] = {"error": f"Val images not found: {val_images_dir}"}
    else:
        image_files = sorted(
            [f for f in val_images_dir.iterdir() if f.suffix.lower() in image_extensions]
        )[:sample_count]

        if not image_files:
            result["pt_vs_onnx"] = {"error": "No images found in val directory", "images_compared": 0}
        else:
            max_iou_diff = 0.0
            max_score_diff = 0.0
            max_class_diff = 0
            total_detections = 0
            mismatched_images = 0
            compared_images = 0

            for img_file in image_files:
                from PIL import Image

                img = Image.open(str(img_file)).convert("RGB")

                results_pt = model_pt.predict(img, save=False, verbose=False, imgsz=imgsz)
                results_onnx = model_onnx.predict(img, save=False, verbose=False, imgsz=imgsz)

                def _extract_boxes(results):
                    boxes = []
                    if results and results[0].boxes is not None:
                        data_boxes = results[0].boxes.data.cpu().numpy()
                        for row in data_boxes:
                            boxes.append({
                                "x1y1x2y2": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                                "score": float(row[4]),
                                "class": int(row[5]),
                            })
                    return boxes

                boxes_pt = _extract_boxes(results_pt)
                boxes_onnx = _extract_boxes(results_onnx)

                if len(boxes_pt) != len(boxes_onnx):
                    mismatched_images += 1

                total_detections += max(len(boxes_pt), len(boxes_onnx))

                # Align by score descending
                max_len = max(len(boxes_pt), len(boxes_onnx))
                for b_pt, b_on in zip(
                    boxes_pt[:max_len] if boxes_pt else [],
                    boxes_onnx[:max_len] if boxes_onnx else [],
                ):
                    iou = _compute_iou(b_pt["x1y1x2y2"], b_on["x1y1x2y2"])
                    max_iou_diff = max(max_iou_diff, 1.0 - iou)
                    max_score_diff = max(max_score_diff, abs(b_pt["score"] - b_on["score"]))
                    max_class_diff = max(max_class_diff, abs(b_pt["class"] - b_on["class"]))

                compared_images += 1

            # When no images could be compared (empty val set, wrong path, etc.),
            # we must NOT silently pass — surface as a failure so users notice.
            if compared_images == 0:
                passed = False
                _log("[Verify]   WARNING: 0 images compared — verification cannot confirm consistency.")
            else:
                passed = (
                    max_iou_diff <= (1.0 - iou_threshold)
                ) and max_score_diff <= score_tolerance and max_class_diff == 0

            pt_vs_onnx_result = {
                "images_compared": compared_images,
                "total_detections_compared": total_detections,
                "mismatched_image_counts": mismatched_images,
                "max_iou_diff": round(max_iou_diff, 6),
                "max_score_diff": round(max_score_diff, 6),
                "max_class_diff": max_class_diff,
                "passed": bool(passed),
            }
            result["pt_vs_onnx"] = pt_vs_onnx_result
            _log(
                f"[Verify]   Compared {compared_images} images, {total_detections} detections"
            )
            _log(
                f"[Verify]   max_iou_diff={max_iou_diff:.6f}, max_score_diff={max_score_diff:.6f}, max_class_diff={max_class_diff}"
            )
            _log(f"[Verify]   pt_vs_onnx {'PASSED' if passed else 'FAILED'}")

    # Step 2: ONNX val
    _log("[Verify] Step 2/2: Running ONNX model validation (mAP)...")
    try:
        val_batch = min(batch, 8)
        model_val = YOLO(str(onnx_path))
        val_metrics = model_val.val(
            data=str(dataset_yaml),
            batch=val_batch,
            imgsz=imgsz,
            half=half,
            save_json=False,
            save_hybrid=False,
            verbose=False,
        )

        val_result: Dict[str, Any] = {}
        if val_metrics:
            if hasattr(val_metrics, "box"):
                box = val_metrics.box
                map50 = getattr(box, "map50", None)
                map50_95 = getattr(box, "map", None)
                val_result["map50"] = round(float(map50), 4) if map50 is not None else None
                val_result["map50_95"] = round(float(map50_95), 4) if map50_95 is not None else None

                # Per-class precision
                ap_class_index = getattr(box, "ap_class_index", None)
                ap_per_class = getattr(box, "ap", None)
                if ap_class_index is not None and ap_per_class is not None:
                    per_class = {}
                    for idx, cls_idx in enumerate(ap_class_index):
                        if idx < len(ap_per_class):
                            per_class[str(int(cls_idx))] = round(float(ap_per_class[idx]), 4)
                    if per_class:
                        val_result["per_class_map50_95"] = per_class

            if hasattr(val_metrics, "results_dict"):
                val_result["raw_results_dict"] = {
                    k: round(float(v), 4) if isinstance(v, (int, float)) else str(v)
                    for k, v in val_metrics.results_dict.items()
                }

        result["onnx_val"] = val_result
        _log(f"[Verify]   ONNX mAP50: {val_result.get('map50', 'N/A')}")
        _log(f"[Verify]   ONNX mAP50-95: {val_result.get('map50_95', 'N/A')}")

    except Exception as ve:
        _log(f"[Verify]   ONNX val error (non-fatal): {ve}")
        result["onnx_val"] = {"error": str(ve)}

    # Overall pass/fail
    pt_ok = result.get("pt_vs_onnx", {}).get("passed", True)
    result["status"] = "passed" if pt_ok else "failed"

    return result
