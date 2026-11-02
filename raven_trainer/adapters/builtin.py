"""Built-in adapters — no entry_points or pip plugins needed.
These adapters are shipped directly with raven-trainer and work
out of the box when the underlying library (ultralytics, etc.) is installed.
This avoids the need for separate raven-ultralytics / raven-mmdetection packages
for the common case, while keeping the adapter interface consistent.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from .base import ModelAdapter, ModelInfo, ModelTask

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?(?:\x07|\x1b\\)|\r")


class _LogCapture(io.TextIOBase):
    """Capture stdout/stderr writes, forward to original AND log callback.

    Strips ANSI escape codes before logging to keep output clean.
    """

    def __init__(
        self,
        orig_stream,
        log_func: Optional[Callable[[str], None]] = None,
        exp_id: str = "",
    ):
        self.orig_stream = orig_stream
        self.log_func = log_func
        self.exp_id = exp_id
        self._buffer = ""

    def write(self, text):
        self.orig_stream.write(text)
        self.orig_stream.flush()
        self._buffer += text
        # Process complete lines (ending with newline or carriage return)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)
        # Also flush on carriage return (progress bar overwrite)
        if self._buffer.endswith("\r"):
            self._emit_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def _emit_line(self, raw_line: str):
        clean = _ANSI_RE.sub("", raw_line).strip()
        if not clean:
            return
        try:
            if self.log_func:
                self.log_func(clean)
        except Exception:
            pass

    def flush(self):
        self.orig_stream.flush()
        if self._buffer.strip():
            self._emit_line(self._buffer)
            self._buffer = ""

    def isatty(self):
        return False




def _validate_dataset_yaml(data_cfg: str, log_func=None):
    """Validate dataset.yaml before passing to ultralytics.
    Checks referenced image paths exist and gives a clear error message.
    """
    import yaml
    _log = log_func or (lambda msg: None)

    yaml_path = Path(data_cfg)
    if not yaml_path.exists():
        raise FileNotFoundError("dataset.yaml not found: " + data_cfg)

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_path = data.get("path", "")
    if not base_path:
        _log("[Validate] dataset.yaml has no 'path' field, ultralytics will resolve relative to yaml location")
        return

    base = Path(base_path)
    if not base.is_absolute():
        base = yaml_path.parent / base_path

    if not base.exists():
        raise FileNotFoundError(
            "Dataset base path not found: " + str(base) + "\n"
            "  Referenced in: " + data_cfg + "\n"
            "  Update the path field in the YAML, or run:\n"
            "    raven-train dataset build [name] --val-size 0.2\n"
            "  to generate a new dataset split."
        )
    for split_name in ["train", "val"]:
        rel = data.get(split_name, "")
        if not rel:
            continue
        img_dir = base / rel / "images"
        if not img_dir.exists():
            _log("[Validate] WARNING: " + split_name + " images directory not found: " + str(img_dir))
        else:
            img_count = len(list(img_dir.glob("*")))
            _log("[Validate] " + split_name + ": " + str(img_dir) + " (" + str(img_count) + " files)")

class UltralyticsAdapter(ModelAdapter):
    """Builtin adapter for ultralytics (YOLO) models.

    Works when `pip install ultralytics` is installed directly,
    without requiring `raven-ultralytics` entry_point package.

    Supports the full ultralytics training hyperparameter surface,
    including all augmentation, loss, and runtime parameters.
    Provides log capture, cancellation support, and ONNX export
    with consistency verification.
    """

    backend_name = "ultralytics"

    # Map non-standard config keys to their valid YOLO equivalents.
    _KEY_ALIASES = {
        "lr": "lr0",                # learning rate -> initial lr
    }

    # Full set of valid Ultralytics YOLO argument keys.
    # Covers detection / segmentation / pose / classification tasks.
    _VALID_YOLO_KEYS: frozenset = frozenset({
        # Data
        "data", "model",
        # Optimization
        "epochs", "batch", "imgsz", "device",
        "optimizer", "lr0", "lrf",
        "weight_decay", "momentum",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "cos_lr", "patience", "seed",
        # Augmentation — Color
        "hsv_h", "hsv_s", "hsv_v",
        # Augmentation — Geometry
        "degrees", "perspective", "scale", "shear", "translate",
        "fliplr", "flipud",
        # Augmentation — Advanced
        "mosaic", "mixup", "copy_paste", "copy_paste_mode",
        "auto_augment", "erasing", "crop_fraction", "cutmix",
        # Loss weights
        "box", "cls", "dfl", "pose", "kobj",
        "cls_pw", "label_smoothing",
        # Runtime
        "workers", "close_mosaic",
        "amp", "deterministic",
        "single_cls", "multi_scale", "rect",
        "dropout", "overlap_mask", "mask_ratio",
        "nbs", "fraction", "freeze",
        "val", "plots", "save", "save_period",
        "exist_ok", "verbose",
        # Task
        "task", "mode",
        # Export
        "format", "opset", "simplify", "dynamic", "half",
        "int8", "keras", "optimize",
        # Inference
        "conf", "iou", "max_det", "agnostic_nms",
        "classes", "augment", "visualize",
        "save_txt", "save_conf", "save_crop", "save_json", "save_frames",
        "stream_buffer", "vid_stride", "line_width",
        "show_labels", "show_conf", "show_boxes",
        # Advanced
        "cache", "cfg", "resume", "pretrained",
        "source", "split", "project", "name",
        "exist_ok", "verbose",
        "nms", "retina_masks", "embed",
        "rle", "profile", "time", "tracker",
        "workspace", "dis", "distill_model",
        "end2end", "max_channels",
        "dnn", "half", "dynamic",
        "simplify", "opset",
        "format", "keras", "optimize", "int8",
        "batch", "imgsz", "device",
        "workers",
    })

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self._cancelled = threading.Event()
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self._log_func: Optional[Callable[[str], None]] = None

    def set_log_func(self, log_func: Optional[Callable[[str], None]]):
        self._log_func = log_func

    def cancel(self):
        """Signal training to cancel at next epoch boundary."""
        self._cancelled.set()

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
                "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            ],
            description=(
                "Ultralytics YOLO models for detection, segmentation, pose, OBB. "
                "Supports full hyperparameter surface including augmentation, "
                "loss weights, and runtime controls."
            ),
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to a concrete device string."""
        if not device or device.lower() != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except ImportError:
            pass
        return "cpu"

    def _filter_yolo_kwargs(self, config: dict) -> dict:
        """Remap aliased keys and drop keys unknown to Ultralytics YOLO."""
        out: Dict[str, Any] = {}
        for k, v in config.items():
            resolved = self._KEY_ALIASES.get(k, k)
            if resolved is None:
                continue  # explicitly dropped
            if resolved in self._VALID_YOLO_KEYS:
                out[resolved] = v
        return out

    def _on_epoch_end(self, trainer):
        """Callback fired at end of each training epoch.
        Checks cancellation signal and raises KeyboardInterrupt if set.
        """
        if self._cancelled.is_set():
            raise KeyboardInterrupt("Training cancelled by user")

    def _extract_metrics(self, results, captured_stdout: str) -> Dict[str, Any]:
        """Extract metrics from training results and captured stdout."""
        metrics: Dict[str, Any] = {}

        # 1. From results.results_dict (best source)
        if hasattr(results, "results_dict") and results.results_dict:
            metrics.update(results.results_dict)

        # 2. From results.metrics
        if hasattr(results, "metrics") and results.metrics:
            m = results.metrics
            if isinstance(m, dict):
                metrics.update(m)
            elif hasattr(m, "box"):
                box = m.box
                if hasattr(box, "map50"):
                    metrics["mAP50"] = float(box.map50)
                if hasattr(box, "map"):
                    metrics["mAP50-95"] = float(box.map)
                if hasattr(box, "precision"):
                    metrics["precision"] = float(box.p)
                if hasattr(box, "recall"):
                    metrics["recall"] = float(box.r)

        # 3. From model.metrics
        if hasattr(self, "_model") and self._model is not None:
            try:
                m = self._model.metrics
                if isinstance(m, dict):
                    metrics.update(m)
            except Exception:
                pass

        # 4. Parse from captured stdout for val metrics at final epoch
        if not metrics:
            for line in captured_stdout.splitlines():
                if "mAP50" in line and "Instances" in line:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            metrics["mAP50"] = float(parts[-3])
                            metrics["mAP50-95"] = float(parts[-2])
                        except (ValueError, IndexError):
                            pass
                        break

        # 5. Parse results.csv for per-epoch metrics
        try:
            import pandas as pd
            run_dirs = list(Path(self.config.get("project", "runs")).iterdir())
            if run_dirs:
                # Find the latest run directory
                csv_paths = []
                for d in sorted(run_dirs, reverse=True):
                    csv = d / "results.csv"
                    if csv.exists():
                        csv_paths.append(csv)
                if csv_paths:
                    df = pd.read_csv(csv_paths[0])
                    if len(df) > 0:
                        last_row = df.iloc[-1].to_dict()
                        metrics["final_epoch"] = int(last_row.get("epoch", 0))
                        for col in df.columns:
                            if col != "epoch":
                                try:
                                    metrics[f"final_{col.strip()}"] = float(last_row[col])
                                except (ValueError, TypeError):
                                    pass
        except Exception:
            pass

        return metrics

    def train(
        self,
        dataset_cfg: str | Path,
        *,
        log_func: Optional[Callable[[str], None]] = None,
        metric_func: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        **kwargs,
    ) -> Any:
        """Run YOLO training with full ultralytics hyperparameter support.

        Args:
            dataset_cfg: Path to dataset.yaml
            log_func: Optional callback for training log messages
            metric_func: Optional callback for per-epoch structured metrics.
                Receives a dict with epoch/loss/mAP/lr keys.
            cancel_event: Optional threading.Event to signal cancellation
            **kwargs: Additional training parameters (overrides config)

        Returns:
            Training results object with .results_dict / .metrics
        """
        from ultralytics import YOLO
        import time as _time

        self._log_func = log_func
        if cancel_event is not None:
            self._cancelled = cancel_event

        def _log(msg: str):
            if log_func:
                log_func(msg)

        model_name = self.config.get("model", kwargs.pop("model", "yolo26n.pt"))
        _log(f"[UltralyticsAdapter] Loading model: {model_name}")
        model = YOLO(model_name)

        # Validate dataset.yaml before passing to ultralytics
        _validate_dataset_yaml(str(dataset_cfg), _log)

        # Merge config and kwargs, kwargs take precedence
        merged_config = dict(self.config)
        merged_config.update(kwargs)

        epochs = merged_config.pop("epochs", 100)
        batch = merged_config.pop("batch", 16)
        imgsz = merged_config.pop("imgsz", 640)
        device = self._resolve_device(
            merged_config.pop("device", "cuda:0")
        )

        # Filter out internal keys before passing to model.train()
        project = merged_config.pop("project", "")
        name = merged_config.pop("name", "train")
        task = merged_config.pop("task", None)

        # Build extra kwargs from remaining config
        extra_kwargs = self._filter_yolo_kwargs(merged_config)

        _log(
            f"[Train] data={dataset_cfg}, epochs={epochs}, batch={batch}, "
            f"imgsz={imgsz}, device={device}, name={name}"
        )
        if extra_kwargs:
            _log(f"[Train] Extra kwargs: {json.dumps(extra_kwargs, default=str)}")

        # Register cancellation callback
        model.add_callback("on_train_epoch_end", self._on_epoch_end)

        # P2: 注册 per-epoch 指标回调，推送结构化 JSON（loss/mAP/lr）
        def _on_fit_epoch_end(trainer):
            """每个 epoch 训练+验证结束后触发，提取结构化指标。"""
            if metric_func is None:
                return
            try:
                raw = getattr(trainer, "metrics", {}) or {}
                epoch = int(getattr(trainer, "epoch", 0))
                # 提取关键字段，归一化 key
                metric_data: Dict[str, Any] = {"epoch": epoch}
                for k, v in raw.items():
                    try:
                        metric_data[k] = float(v) if hasattr(v, "item") else v
                    except Exception:
                        metric_data[k] = v
                metric_func(metric_data)
            except Exception:
                pass  # 指标推送失败不影响训练

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        self._model = model

        # Capture stdout/stderr during training
        _old_stdout = sys.stdout
        _old_stderr = sys.stderr
        _stdout_buf = io.StringIO()

        _stdout_cap = _LogCapture(_old_stdout, log_func)
        _stderr_cap = _LogCapture(_old_stderr, log_func)

        # Also capture to string buffer for metrics extraction
        class _DualCapture:
            def __init__(self, cap, buf):
                self.cap = cap
                self.buf = buf

            def write(self, text):
                self.cap.write(text)
                self.buf.write(text)

            def flush(self):
                self.cap.flush()
                self.buf.flush()

            def isatty(self):
                return False

        _dual_out = _DualCapture(_stdout_cap, _stdout_buf)
        _dual_err = _DualCapture(_stderr_cap, _stdout_buf)

        try:
            sys.stdout = _dual_out
            sys.stderr = _dual_err

            _log("[Train] Starting training...")
            results = model.train(
                data=str(dataset_cfg),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project=project,
                name=name,
                task=task,
                exist_ok=True,
                verbose=True,
                **extra_kwargs,
            )
            _log("[Train] Training completed successfully!")

        except KeyboardInterrupt:
            _log("[Train] Training cancelled by user.")
            # Re-raise so caller can handle
            raise

        except Exception as e:
            _log(f"[Train] Training FAILED: {e}")
            _log(traceback.format_exc())
            raise

        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
            # Always remove callbacks to avoid leak on re-use of the model instance
            try:
                model.remove_callback("on_train_epoch_end", self._on_epoch_end)
                model.remove_callback("on_fit_epoch_end", _on_fit_epoch_end)
            except Exception:
                pass
            self._model = None

        _captured = _stdout_buf.getvalue()

        # Extract metrics
        metrics = self._extract_metrics(results, _captured)
        if metrics:
            _log(f"[Train] Metrics: {json.dumps(metrics, default=str)}")

        return results

    def evaluate(
        self,
        dataset_cfg: str | Path,
        *,
        weights: Optional[str] = None,
        split: str = "val",
        device: str = "cuda:0",
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        log_func: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a trained model on a dataset split using ultralytics val.

        Returns the metrics dict (mAP, per-class metrics, etc.). Returns empty
        dict only if the underlying val call yields nothing.
        """
        from ultralytics import YOLO

        _log = log_func or (lambda msg: None)

        pt_path = Path(weights or self.config.get("weights") or self.config.get("model", ""))
        if not pt_path or not Path(pt_path).exists():
            raise FileNotFoundError(f"Weights not found: {pt_path}")

        _log(f"[Eval] Loading model: {pt_path}")
        model = YOLO(str(pt_path))

        val_args: Dict[str, Any] = {
            "data": str(dataset_cfg),
            "split": split,
            "device": self._resolve_device(device),
        }
        if conf is not None:
            val_args["conf"] = conf
        if imgsz is not None:
            val_args["imgsz"] = imgsz
        if batch is not None:
            val_args["batch"] = batch
        # Forward any additional ultralytics kwargs (e.g. iou, project, name)
        val_args.update(self._filter_yolo_kwargs(kwargs))

        _log(f"[Eval] Running val on split='{split}' device='{device}'...")
        try:
            results = model.val(**val_args)
        except Exception as e:
            _log(f"[Eval] val FAILED: {e}")
            _log(traceback.format_exc())
            raise

        # Ultralytics results object exposes a .results_dict attribute
        metrics: Dict[str, Any] = {}
        try:
            if hasattr(results, "results_dict"):
                metrics = dict(results.results_dict)
            elif isinstance(results, dict):
                metrics = dict(results)
            else:
                # Fallback: pull common attributes
                for attr in ("box.map", "box.map50", "box.map75"):
                    try:
                        val = float(results.__getattr__(attr))
                        metrics[attr] = val
                    except Exception:
                        pass
        except Exception:
            pass

        # Coerce numpy scalars to float for JSON serialisation
        cleaned: Dict[str, Any] = {}
        for k, v in metrics.items():
            try:
                cleaned[k] = float(v) if hasattr(v, "item") else v
            except Exception:
                cleaned[k] = v

        _log(f"[Eval] {len(cleaned)} metrics returned.")
        return cleaned

    def export(
        self,
        output_path: str | Path,
        fmt: str = "onnx",
        *,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        imgsz: int = 640,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 17,
        dataset_yaml: Optional[str] = None,
        verify: bool = False,
        log_func: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export trained model with optional consistency verification.

        Args:
            output_path: Output file path for exported model
            fmt: Export format (onnx, tensorrt, etc.)
            model_path: Path to .pt model file. If None, uses config.
            device: Export device
            imgsz: Input image size
            half: FP16 half precision
            int8: INT8 quantization (requires dataset_yaml for calibration)
            dynamic: Dynamic axes (batch/h/w)
            simplify: ONNX simplify (onnxsim)
            opset: ONNX opset version
            dataset_yaml: Path to dataset.yaml (required for verify=True and int8=True)
            verify: Run pt-vs-onnx consistency check + val
            log_func: Optional log callback
            **kwargs: Additional export parameters

        Returns:
            Dict with export results including optional verification data
        """
        from ultralytics import YOLO

        _log = log_func or (lambda msg: None)

        pt_path = Path(model_path or self.config.get("model", ""))
        if not pt_path.exists():
            raise FileNotFoundError(f"Model not found: {pt_path}")

        _log(f"[Export] Loading model: {pt_path}")
        model = YOLO(str(pt_path))

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_args: Dict[str, Any] = {
            "format": fmt,
            "imgsz": imgsz,
            "device": device,
            "opset": opset,
            "simplify": simplify,
            "dynamic": dynamic,
        }
        if half:
            export_args["half"] = True
        if int8:
            # INT8 静态量化需要校准数据集（ultralytics 用 data 参数指定）
            if dataset_yaml:
                export_args["int8"] = True
                export_args["data"] = str(dataset_yaml)
                _log("[Export] INT8 quantization enabled with calibration dataset")
            else:
                _log("[Export] WARNING: INT8 requested but no dataset_yaml provided for calibration, skipping int8")
                int8 = False

        _log(
            f"[Export] fmt={fmt}, imgsz={imgsz}, device={device}, "
            f"opset={opset}, half={half}, int8={int8}, dynamic={dynamic}, simplify={simplify}"
        )

        orig_cwd = os.getcwd()
        os.chdir(str(out_path.parent))

        result: Dict[str, Any] = {}
        try:
            model.export(**export_args)

            # Locate exported file
            onnx_name = pt_path.stem + ".onnx"
            onnx_path = out_path.parent / onnx_name
            if not onnx_path.exists():
                onnx_path = out_path  # assume the provided path
            if not onnx_path.exists():
                for f in out_path.parent.rglob("*.onnx"):
                    onnx_path = f
                    break

            if not onnx_path.exists():
                raise FileNotFoundError("ONNX file not found after export")

            file_size = onnx_path.stat().st_size
            result = {
                "onnx_path": str(onnx_path.resolve()),
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "imgsz": imgsz,
                "half": half,
                "int8": int8,
                "dynamic": dynamic,
                "simplify": simplify,
                "opset": opset,
            }
            _log("[Export] ONNX saved: {onnx_path} ({file_size / 1024 / 1024:.1f} MB)")

            # Optional consistency verification
            if verify and dataset_yaml:
                from .onnx_verify import verify_onnx_consistency
                _log("[Export] Running consistency verification...")
                try:
                    verify_result = verify_onnx_consistency(
                        pt_path=str(pt_path),
                        onnx_path=str(onnx_path),
                        dataset_yaml=dataset_yaml,
                        imgsz=imgsz,
                        half=half,
                        log_func=_log,
                    )
                    result["verification"] = verify_result
                    _log(
                        f"[Export] Verification: {verify_result.get('status', 'unknown')}"
                    )
                except Exception as ve:
                    _log(f"[Export] Verification error (non-fatal): {ve}")
                    result["verification"] = {"status": "error", "error": str(ve)}

        finally:
            os.chdir(orig_cwd)

        return result


# Registry of builtin adapters (name -> class)
BUILTIN_ADAPTERS: Dict[str, Type[ModelAdapter]] = {
    "ultralytics": UltralyticsAdapter,
}
