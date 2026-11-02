"""RF-DETR ONNX export — vendored from rf-detr-onnx (read-only reference) + local patches.

Reference: ``E:\\study-place\\Davidche\\2026\\rf-detr-onnx\\export.py`` (kept
read-only).  Vendored pieces:
  - ``unwrap_torch_model`` / ``_ForwardExportWrapper`` / ``get_resolution``
  - the torch.onnx.export flow with dynamic axes + onnxsim

Local patches vs. the reference (P1-9):
  - antialias ``F.interpolate`` patch is applied INSIDE the export function in
    a try/finally and restored afterwards — the reference patches it at import
    time (global side effect that would poison other backends in the same
    process).
  - self-developed metadata (model_name/model_version/labels/sha256/resolution
    + normalization constants) written into the ONNX model.
  - TRUE dynamic batch: the model is switched to export mode
    (``LWDETR.export()`` → ``forward_export``) before tracing. The default
    ``forward`` wraps the input in a NestedTensor
    (``nested_tensor_from_tensor_list`` indexes ``tensor_list[0]``), so tracing
    bakes a batch-1 ``Split(axis=0, sizes=[1]) + Squeeze`` into the ONNX — the
    declared ``dynamic_axes`` then lies and any batch>1 run crashes. Neither
    the rf-detr reference exporter (``rf-detr-onnx/export.py``) nor the
    upstream ``rfdetr/export/main.py`` call ``export()``; both ship this
    batch-lock. ``forward_export`` keeps the batch dim through the whole graph
    (all-zero padding masks, no NestedTensor indexing), making the exported
    ONNX    genuinely batch-dynamic.
  - output names are derived from the model heads (``segmentation_head`` /
    ``use_grouppose_keypoints``) instead of the vendored shape heuristic: seg
    masks (B,N,H,W) and keypoints (B,N,K,2) are both 4-D, so the reference's
    ``_infer_output_names`` would mislabel seg masks as ``pred_keypoints``.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import labels_from_dataset
from .onnx_contract import (
    INPUT_NAME,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    OUTPUT_BOXES,
    OUTPUT_KEYPOINTS,
    OUTPUT_LOGITS,
    OUTPUT_MASKS,
)

# torch / onnx / onnxsim 均为重依赖，惰性导入（保证 import raven_rfdetr
# 与最小测试骨架不要求安装导出依赖）
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_onnx = None
_onnxsim = None


def _load_onnx():
    global _onnx
    if _onnx is None:
        import onnx
        _onnx = onnx
    return _onnx


def _load_onnxsim():
    global _onnxsim
    if _onnxsim is None:
        from onnxsim import simplify
        _onnxsim = simplify
    return _onnxsim


# ---------------------------------------------------------------------------
# Vendored helpers (from rf-detr-onnx/export.py)
# ---------------------------------------------------------------------------

def unwrap_torch_model(model_wrapper: Any) -> torch.nn.Module:
    """Resolve the bare nn.Module inside an RFDETR wrapper."""
    candidates = ["model.model", "model"]
    for path in candidates:
        obj = model_wrapper
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.Module):
                return obj
        except Exception:
            pass
    raise RuntimeError("Unable to resolve torch model from RFDETR wrapper")


def get_resolution(model_wrapper: Any) -> int:
    try:
        return int(model_wrapper.model_config.resolution)
    except Exception:
        return 640


class _ForwardExportWrapper(torch.nn.Module):
    """Return only inference-relevant outputs (no aux/enc bloat)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        results = [out[OUTPUT_BOXES], out[OUTPUT_LOGITS]]
        if "pred_keypoints" in out:
            results.append(out["pred_keypoints"])
        elif OUTPUT_MASKS in out:
            results.append(out[OUTPUT_MASKS])
        return tuple(results)


def _infer_output_names(outputs) -> List[str]:
    """Vendored shape-based fallback (only used when model attrs are
    inconclusive — see ``_infer_output_names_from_model``).

    NOTE: the 4-D heuristic below is ambiguous — seg masks are (B,N,H,W)
    and keypoints are (B,N,K,2), both 4-D.  Prefer the head-based inference.
    """
    if isinstance(outputs, dict):
        return list(outputs.keys())
    if isinstance(outputs, (list, tuple)):
        num = len(outputs)
        if num == 3:
            third = outputs[2]
            if isinstance(third, torch.Tensor) and third.dim() == 4:
                return [OUTPUT_BOXES, OUTPUT_LOGITS, "pred_keypoints"]
            return [OUTPUT_BOXES, OUTPUT_LOGITS, OUTPUT_MASKS]
        if num == 2:
            return [OUTPUT_BOXES, OUTPUT_LOGITS]
        return [f"output_{i}" for i in range(num)]
    return ["output"]


def _infer_output_names_from_model(model: torch.nn.Module) -> List[str]:
    """Derive ONNX output names from the model's heads — exact, no shape
    guessing.  Seg masks (B,N,H,W) and keypoints (B,N,K,2) are both 4-D, so
    the vendored shape heuristic can't tell them apart; the model exposes the
    truth via ``segmentation_head`` / ``use_grouppose_keypoints``.
    ``segmentation_head`` is checked first to match ``forward_export``'s
    return precedence (masks win when both heads exist).
    """
    names = [OUTPUT_BOXES, OUTPUT_LOGITS]
    if getattr(model, "segmentation_head", None) is not None:
        names.append(OUTPUT_MASKS)
    elif getattr(model, "use_grouppose_keypoints", False):
        names.append(OUTPUT_KEYPOINTS)
    return names


# ---------------------------------------------------------------------------
# antialias patch (local, restored afterwards)
# ---------------------------------------------------------------------------

_original_interpolate = F.interpolate


def _patched_interpolate(*args, **kwargs):
    """ONNX does not support aten::_upsample_bicubic2d_aa."""
    if "antialias" in kwargs:
        kwargs["antialias"] = False
    return _original_interpolate(*args, **kwargs)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _embed_metadata(
    onnx_path: Path,
    model_name: str,
    model_version: str,
    labels: List[str],
    resolution: int,
    mean: List[float],
    std: List[float],
) -> None:
    """Write self-developed metadata into the ONNX model (P2-18)."""
    onnx = _load_onnx()
    model = onnx.load(str(onnx_path))
    props: Dict[str, str] = {
        "model_name": model_name,
        "model_version": model_version,
        "labels": ",".join(labels),
        "sha256": _sha256(onnx_path),
        "resolution": str(resolution),
        # 归一化在 resolution 内完成（ImageNet mean/std，非 YOLO letterbox）
        "normalization_mean": ",".join(f"{m:.6f}" for m in mean),
        "normalization_std": ",".join(f"{s:.6f}" for s in std),
        # 坐标格式：xyxy（归一化）→ 下游按原图尺寸缩放；输出顺序 boxes→logits[→masks]
        "coordinate_format": "xyxy_normalized",
        "output_order": f"{OUTPUT_BOXES},{OUTPUT_LOGITS},{OUTPUT_MASKS}",
        "framework": "rfdetr",
    }
    existing = {p.key: p.value for p in model.metadata_props}
    existing.update(props)
    # 替换而非追加，避免重复
    del model.metadata_props[:]
    for k, v in existing.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v
    onnx.save(model, str(onnx_path))


# ---------------------------------------------------------------------------
# main export flow
# ---------------------------------------------------------------------------

def export_rfdetr_onnx(
    output_path: str,
    model_path: str,
    model: Any = None,
    device: str = "cpu",
    resolution: Optional[int] = None,
    dynamic_batch: bool = True,
    simplify: bool = True,
    opset: int = 17,
    half: bool = False,
    dataset_yaml: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    log_func: Optional[Callable[[str], None]] = None,
    verify: bool = False,
    adapter: Any = None,
) -> Dict[str, Any]:
    """Export an RF-DETR checkpoint to ONNX (vendored flow + local patches).

    ``model_path`` must be a full training checkpoint (``args`` + ``model``).
    For weights-only state dicts (``{"model": ...}`` — e.g. the published
    ``rf-detr-seg-*.pt`` / ``rf-detr-keypoint-*.pth`` starter weights, which
    carry no ``args``), pass a pre-built RFDETR facade via ``model`` instead.

    Returns dict with ``onnx_path`` (+ ``verification`` when ``verify=True``).
    """
    _log = log_func or (lambda msg: None)
    metadata = metadata or {}

    if model is not None:
        rfdetr_model = model
    else:
        from rfdetr.detr import RFDETR

        _log(f"[RFDETR] Loading model: {model_path}")
        rfdetr_model = RFDETR.from_checkpoint(model_path)

    torch_model = unwrap_torch_model(rfdetr_model)
    torch_model.eval()
    torch_model.cpu()

    res = resolution or get_resolution(rfdetr_model)
    _log(f"[RFDETR] Using resolution: {res}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, res, res, dtype=torch.float32, device="cpu")

    # 局部 antialias patch（try/finally 恢复，避免进程级副作用）
    _log("[RFDETR] Applying local antialias patch (restored after export)...")
    F.interpolate = _patched_interpolate

    # 真动态 batch：优先切 export 模式（LWDETR.export() → forward_export，全图
    # 保留 batch 维）。仅当模型不支持时回退到 forward + _ForwardExportWrapper
    # （该路径追踪时会烘焙 batch=1 的 Split/Squeeze，dynamic_axes 名不副实）。
    use_export_mode = hasattr(torch_model, "export") and hasattr(torch_model, "forward_export")
    if use_export_mode:
        _log("[RFDETR] Switching model to export mode (forward_export) for true dynamic batch...")
        torch_model.export()
    else:
        _log("[RFDETR] Model has no export mode; falling back to forward tracing (batch locked to 1)")

    try:
        with torch.no_grad():
            if use_export_mode:
                outputs = torch_model.forward_export(dummy_input)
            else:
                wrapper = _ForwardExportWrapper(torch_model)
                outputs = wrapper(dummy_input)
        # 优先按模型头推断输出名（seg 的 mask 与 keypoints 同为 4-D，形状启发
        # 式无法区分）；只有头信息不完整时才回退到 vendored 形状启发式。
        output_names = _infer_output_names_from_model(torch_model)
        if len(output_names) != len(outputs):
            output_names = _infer_output_names(outputs)
        _log(f"[RFDETR] Output names: {output_names}")

        dynamic_axes: Dict[str, Dict[int, str]] = {INPUT_NAME: {0: "batch", 2: "height", 3: "width"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                torch_model if use_export_mode else wrapper,
                dummy_input,
                str(out_path),
                export_params=True,
                opset_version=max(opset, 17),
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                input_names=[INPUT_NAME],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                dynamo=False,
            )

        onnx = _load_onnx()
        _log("[RFDETR] Validating ONNX model...")
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

        if simplify:
            _log("[RFDETR] Simplifying ONNX model (onnxsim)...")
            simplify_fn = _load_onnxsim()
            model_simplified, check = simplify_fn(onnx_model)
            if not check:
                raise RuntimeError("ONNX simplification failed")
            onnx.save(model_simplified, str(out_path))
    finally:
        F.interpolate = _original_interpolate

    # 自研 metadata 写入（P1-9/P2-18）
    model_name = metadata.get("model_name") or Path(model_path).stem
    model_version = metadata.get("model_version") or "1.0"
    labels = metadata.get("labels")
    if not labels and dataset_yaml:
        labels = labels_from_dataset(dataset_yaml)
    labels = labels or []
    _log(f"[RFDETR] Embedding metadata: model={model_name} v{model_version} "
         f"labels={labels}")
    _embed_metadata(
        out_path,
        model_name=model_name,
        model_version=model_version,
        labels=labels,
        resolution=res,
        mean=list(NORMALIZATION_MEAN),
        std=list(NORMALIZATION_STD),
    )

    file_size = out_path.stat().st_size
    result: Dict[str, Any] = {
        "onnx_path": str(out_path.resolve()),
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "resolution": res,
        "dynamic": dynamic_batch,
        "simplify": simplify,
        "opset": max(opset, 17),
        "half": half,
        "labels": labels,
    }
    _log(f"[RFDETR] ONNX saved: {out_path} ({file_size / 1024 / 1024:.1f} MB)")

    if verify:
        from .verify import verify_pt_vs_onnx
        _log("[RFDETR] Running pt-vs-onnx consistency verification...")
        try:
            result["verification"] = verify_pt_vs_onnx(
                pt_path=model_path,
                onnx_path=str(out_path),
                dataset_yaml=dataset_yaml,
                resolution=res,
                log_func=_log,
                pt_model=model,  # 复用传入的预构建模型（权重-only state dict 无 args）
            )
        except Exception as ve:
            _log(f"[RFDETR] Verification error (non-fatal): {ve}")
            result["verification"] = {"status": "error", "error": str(ve)}

    return result
