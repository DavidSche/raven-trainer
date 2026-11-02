"""Raven Trainer — RF-DETR backend adapter.

Wraps the ``rfdetr`` package (``RFDETR<Size>`` classes + PyTorch Lightning
training stack) behind the platform ``ModelAdapter`` contract.  Registered via
entry_points group ``raven_trainer.adapters`` (backend name ``rfdetr``).

Functional baseline: ``raven_trainer.adapters.builtin.UltralyticsAdapter`` —
``export()`` MUST return a dict (at least ``onnx_path``), ``train()`` accepts
``log_func``/``metric_func``/``cancel_event``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from raven_trainer.adapters.base import ModelAdapter, ModelInfo, ModelTask

from . import config as _config
from .config import (
    MODEL_VARIANTS,
    MODEL_VARIANT_ALIASES,
    ensure_rf_yolo_layout,
    resolve_variant,
    validate_pretrain_weights,
)
from .evaluate import normalize_metrics
from .train import train_rfdetr
from .export import export_rfdetr_onnx

try:
    from .verify import verify_pt_vs_onnx
    _VERIFY_AVAILABLE = True
except Exception:  # pragma: no cover - defensive; verify imports torch/onnx lazily
    _VERIFY_AVAILABLE = False


__all__ = [
    "RavenRFDETRAdapter",
    "MODEL_VARIANTS",
    "MODEL_VARIANT_ALIASES",
]


class RavenRFDETRAdapter(ModelAdapter):
    """Adapter for RF-DETR detection / segmentation models.

    Requires ``rfdetr`` (>= 1.10.0.dev, see pyproject pin) with training extras:
    ``pip install "rfdetr[train,loggers]"``.
    """

    backend_name = "rfdetr"

    @classmethod
    def info(cls) -> ModelInfo:
        return ModelInfo(
            backend="rfdetr",
            task=ModelTask.DETECTION,
            framework="rfdetr",
            supported_models=list(MODEL_VARIANTS.keys()),
            description=(
                "RF-DETR detection/segmentation models (nano/small/base/medium/large "
                "and seg-nano/small/medium/large). Data side uses YOLO layout — "
                "dataset.yaml is projected to data.yaml automatically."
            ),
        )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        try:
            import rfdetr  # noqa: F401
        except ImportError:
            raise ImportError(
                "rfdetr is not installed. Run: pip install -e <rf-detr-src> && "
                'pip install "rfdetr[train,loggers]"'
            )

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def train(
        self,
        dataset_cfg: str,
        *,
        log_func: Optional[Callable[[str], None]] = None,
        metric_func: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run RF-DETR training on a Raven dataset.yaml workspace.

        Returns a CLI-consumable dict: ``{"model_path", "run_dir", "metrics",
        "artifact_paths"}`` (upstream ``RFDETR.train()`` returns ``None``).

        ``cancel_event`` is accepted for interface parity but is NOT wired to
        the upstream training loop (rf-detr exposes no stop/cancel API; see
        plan §6.3 — 一期用 early_stopping + 外部 kill 兜底).
        """
        _log = log_func or (lambda msg: None)
        if cancel_event is not None:
            _log("[RFDETR] NOTE: cancel_event is not supported by rf-detr (no stop API); "
                 "use early_stopping + external kill.")

        dataset_cfg = str(dataset_cfg)
        cfg = dict(self.config)
        cfg.update(kwargs)

        model_name = cfg.pop("model", None)
        variant = resolve_variant(model_name)

        # pretrain_weights 预检（本地路径或官方注册权重名，无 URL）；
        # data.yaml 幂等投影在 train_rfdetr 内部执行
        validate_pretrain_weights(cfg.get("pretrain_weights"))

        result = train_rfdetr(
            dataset_cfg=dataset_cfg,
            variant=variant,
            log_func=_log,
            metric_func=metric_func,
            config=cfg,
        )
        return result

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        dataset_cfg: str,
        *,
        split: str = "val",
        weights: Optional[str] = None,
        device: str = "cuda:0",
        log_func: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a trained RF-DETR checkpoint on a dataset split.

        Normalizes upstream split-prefixed metric keys to Raven conventions
        (``val/mAP_50_95`` -> ``mAP50-95``, ``mAR`` has no suffix, per-class
        ``AP/<name>`` -> ``per_class/<name>``).  Empty dict => CLI FAILED.
        """
        from .evaluate import evaluate_rfdetr

        _log = log_func or (lambda msg: None)
        dataset_cfg = str(dataset_cfg)
        weights = str(weights or self.config.get("weights") or self.config.get("model", ""))
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        ensure_rf_yolo_layout(Path(dataset_cfg).parent, dataset_cfg, log_func=_log)

        raw = evaluate_rfdetr(
            dataset_cfg=dataset_cfg,
            weights=weights,
            split=split,
            device=device,
            log_func=_log,
            **kwargs,
        )
        return normalize_metrics(raw, split=split)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export(
        self,
        output_path: str,
        fmt: str = "onnx",
        *,
        model_path: Optional[str] = None,
        device: str = "cpu",
        imgsz: Optional[int] = None,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = True,
        simplify: bool = True,
        opset: int = 17,
        dataset_yaml: Optional[str] = None,
        verify: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        log_func: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export trained RF-DETR checkpoint to ONNX (vendored rf-detr-onnx flow).

        Returns dict with ``onnx_path`` (+ optional ``verification`` when
        ``verify=True``), aligned with the builtin UltralyticsAdapter contract.
        """
        from .export import export_rfdetr_onnx

        _log = log_func or (lambda msg: None)
        if fmt != "onnx":
            raise ValueError(f"RF-DETR adapter only supports fmt='onnx', got {fmt!r}")
        if int8:
            raise ValueError("RF-DETR adapter does not support int8 quantization yet")

        model_path = str(model_path or self.config.get("weights") or self.config.get("model", ""))
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        result = export_rfdetr_onnx(
            output_path=output_path,
            model_path=model_path,
            device=device,
            resolution=imgsz,
            dynamic_batch=dynamic,
            simplify=simplify,
            opset=opset,
            half=half,
            dataset_yaml=dataset_yaml,
            metadata=metadata,
            log_func=_log,
            verify=verify,
            adapter=self,
        )
        return result

    # ------------------------------------------------------------------
    # artifact paths
    # ------------------------------------------------------------------
    def artifact_paths(self, run_dir: str) -> Dict[str, str]:
        """Locate rf-detr checkpoint artifacts in a training run directory.

        Returns dict with optional keys (only existing files are included):
        ``best_model`` (checkpoint_best_total.pth), ``last_ema`` (last_ema.pth),
        ``last_checkpoint`` (last.ckpt, PTL 全量可 resume), ``metrics_csv``.
        """
        return _artifact_paths_from_run(Path(run_dir))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extract_metrics(self, result: Any, captured_stdout: str = "") -> Dict[str, Any]:
        """Parity shim so cli/train.py can keep using its existing metric path."""
        if isinstance(result, dict):
            return dict(result.get("metrics") or {})
        return {}


def _artifact_paths_from_run(run_dir: Path) -> Dict[str, str]:
    """Shared artifact locator (also used by export path logic)."""
    paths: Dict[str, str] = {}
    candidates = {
        "best_model": run_dir / "checkpoint_best_total.pth",
        "best_ema": run_dir / "checkpoint_best_ema.pth",
        "last_ema": run_dir / "last_ema.pth",
        "last_checkpoint": run_dir / "last.ckpt",
        "metrics_csv": run_dir / "metrics.csv",
    }
    for key, p in candidates.items():
        if p.exists():
            paths[key] = str(p)
    return paths
