"""Raven Trainer — Ultralytics (YOLO) backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from raven_trainer.adapters.base import ModelAdapter, ModelInfo, ModelTask


# YOLO-recognized training kwargs only — anything else is silently dropped
_YOLO_TRAIN_KWARGS = {
    "data", "epochs", "batch", "imgsz", "patience", "device", "workers",
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "pose", "kobj",
    "label_smoothing", "nbs", "overlap_mask", "mask_ratio", "dropout",
    "val", "plots", "save", "save_period", "project", "name", "exist_ok",
    "pretrained", "optimizer", "verbose", "seed", "deterministic",
    "single_cls", "rect", "cos_lr", "close_mosaic", "resume", "amp",
    "fraction", "profile", "cache", "image_weights", "multi_scale",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "auto_augment", "erasing", "crop_fraction",
    # 3 more real ultralytics train kwargs surfaced by the New Training form
    "freeze", "compile", "pin_memory",
}


def _filter_yolo_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in _YOLO_TRAIN_KWARGS}


class RavenUltralyticsAdapter(ModelAdapter):
    """Adapter for Ultralytics YOLO models (detection, pose, seg, cls)."""

    backend_name = "ultralytics"

    @classmethod
    def info(cls) -> ModelInfo:
        return ModelInfo(
            backend="ultralytics",
            task=ModelTask.DETECTION,
            framework="ultralytics",
            supported_models=[
                "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
                "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose",
                "yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg",
                "yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls",
                "yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb", "yolo26x-obb",
                "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
                "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
                "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
                "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
                "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb",
                "rtdetr-l",
            ],
            description="Ultralytics YOLO models",
        )

    def train(self, dataset_cfg: str | Path, **kwargs) -> Any:
        from ultralytics import YOLO

        model_name = self.config.get("model", "yolo26n")
        model = YOLO(model_name)

        train_kwargs = {**self.config, **kwargs}
        train_kwargs["data"] = str(dataset_cfg)
        train_kwargs = _filter_yolo_kwargs(train_kwargs)

        return model.train(**train_kwargs)

    def evaluate(self, dataset_cfg, **kwargs):
        from ultralytics import YOLO
        weights = self.config.get("weights", "")
        model_name = self.config.get("model", "yolo26n")
        model_path = weights if weights and Path(weights).exists() else model_name
        model = YOLO(str(model_path))
        val_kwargs = {"data": str(dataset_cfg), "split": kwargs.get("split", "val"), "device": kwargs.get("device", "cuda:0")}
        if kwargs.get("conf"): val_kwargs["conf"] = kwargs["conf"]
        if kwargs.get("imgsz"): val_kwargs["imgsz"] = kwargs["imgsz"]
        if kwargs.get("batch"): val_kwargs["batch"] = kwargs["batch"]
        results = model.val(**val_kwargs)
        if hasattr(results, "results_dict"):
            return results.results_dict
        return {}

    def export(self, output_path: str | Path, **kwargs) -> Dict[str, Any]:
        """Export trained model.

        P1-4: MUST return a Dict (at least ``onnx_path``), NOT a bare Path —
        aligns with `adapters/base.py`'s contract and the builtin adapter.
        """
        from ultralytics import YOLO

        model = YOLO(self.config.get("model", "yolo26n"))
        fmt = kwargs.get("fmt", "onnx")
        model.export(format=fmt, device=kwargs.get("device", "cuda:0"))

        out = Path(output_path)
        if not out.exists():
            # ultralytics 导出到 runs/<run>/weights/<model>.onnx — 取最新产物
            candidates = sorted(Path("runs").rglob("*.onnx")) if Path("runs").exists() else []
            if candidates:
                out = candidates[-1]
        return {"onnx_path": str(out.resolve())}
