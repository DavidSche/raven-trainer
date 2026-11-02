"""RF-DETR ONNX I/O contract constants (P2-18).

Verified against rf-detr-onnx reference + rfdetr 1.10.0.dev:
- input ``input``: [N, 3, H, W], RGB, ImageNet mean/std normalization
  performed OUTSIDE the graph (raven-go scanner must apply it), square resize
  (NOT YOLO letterbox) to ``resolution``.
- outputs ``pred_boxes`` / ``pred_logits`` [/ ``pred_masks`` | ``pred_keypoints``]
  in that order (boxes first).
"""

from __future__ import annotations

from typing import Tuple

#: ONNX input tensor name
INPUT_NAME = "input"

#: ONNX output tensor names (in order)
OUTPUT_BOXES = "pred_boxes"
OUTPUT_LOGITS = "pred_logits"
OUTPUT_MASKS = "pred_masks"
OUTPUT_KEYPOINTS = "pred_keypoints"

#: Expected output order (boxes first, matches _ForwardExportWrapper)
OUTPUT_ORDER = (OUTPUT_BOXES, OUTPUT_LOGITS, OUTPUT_MASKS, OUTPUT_KEYPOINTS)

#: ImageNet normalization constants (mean/std) — applied by the runtime,
#: NOT embedded in the graph.
NORMALIZATION_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
NORMALIZATION_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Coordinate format: boxes are normalized xyxy (0..1 relative to input),
#: downstream multiplies by original image width/height.
COORDINATE_FORMAT = "xyxy_normalized"

#: Minimum supported opset
MIN_OPSET = 17
