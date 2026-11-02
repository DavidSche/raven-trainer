"""raven-train train — End-to-end model training with Experiment tracking.

Usage:
    # Train from a project (created by 'raven-train new')
    raven-train train projects/person_intrusion/project.yaml

    # Train directly from a workspace dataset.yaml (quick training)
    raven-train train data/workspace/workshop_split/dataset.yaml

Training flow:
    1. Load config (from project.yaml or standalone dataset.yaml)
    2. Load backend adapter (ultralytics, mmdetection, etc.)
    3. Run training with full hyperparameter support
    4. Save experiment record to projects/<name>/experiments/
    5. Export best model to projects/<name>/models/
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from ..adapters import load_adapter
from ..config import R
from ..data_asset.job import JobManager, JobStatus, JobType


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_experiment(
    project_name: str,
    config: Dict[str, Any],
    results: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
) -> Path:
    """Save experiment record to projects/<name>/experiments/exp_<seq>.yaml."""
    exp_dir = R.experiment_dir(project_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(exp_dir.glob("exp_*.yaml"))
    seq = len(existing) + 1
    exp_path = exp_dir / "exp_{seq:03d}.yaml"

    record = {
        "experiment": "exp_{seq:03d}",
        "created_at": datetime.now().isoformat(),
        "config": config,
        "results": results or {},
    }
    if model_path:
        record["best_model"] = str(model_path)

    with open(exp_path, "w", encoding="utf-8") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)

    return exp_path


def _collect_metrics_from_csv(project_dir: Path) -> Dict[str, Any]:
    """Collect metrics from results.csv in the latest run directory."""
    metrics: Dict[str, Any] = {}
    runs_base = project_dir / "runs"
    if not runs_base.exists():
        return metrics
    try:
        import pandas as pd
        csv_paths = sorted(runs_base.rglob("results.csv"))
        if csv_paths:
            df = pd.read_csv(csv_paths[-1])
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


def _find_best_model(project_dir: Path, run_name: Optional[str] = None) -> Optional[Path]:
    """Find the best model checkpoint in the project's runs directory.

    Supports Ultralytics ``weights/best.pt`` layout AND RF-DETR
    ``checkpoint_best_{total,ema}.pth`` layout (P0-3 / P1-6).
    """
    if run_name:
        run_dir = project_dir / "runs" / run_name
        for bp in (
            run_dir / "weights" / "best.pt",
            run_dir / "checkpoint_best_total.pth",
            run_dir / "checkpoint_best_ema.pth",
            run_dir / "checkpoint_best_regular.pth",
            run_dir / "last_ema.pth",
        ):
            if bp.exists():
                return bp
    runs_base = project_dir / "runs"
    if runs_base.exists():
        for runs_dir in sorted(runs_base.iterdir(), reverse=True):
            for bp in (
                runs_dir / "weights" / "best.pt",
                runs_dir / "best.pt",
                runs_dir / "checkpoint_best_total.pth",
                runs_dir / "checkpoint_best_ema.pth",
            ):
                if bp.exists():
                    return bp
    return None


def _do_train(
    config_path,
    device="cuda:0",
    epochs=None,
    batch=None,
    imgsz=None,
    log_func=print,
    metric_func=None,
    export_format="onnx",
    export_opset=17,
    export_half=False,
    export_int8=False,
    # P0-2: 与 EXPORT_SPEC（动态 batch=Yes）对齐，默认动态导出
    export_dynamic=True,
    export_simplify=True,
    cancel_event=None,
    register_to_platform=False,
    hyperparameters=None,
):
    """Programmatic training entry point for API.

    Supports full ultralytics hyperparameter passthrough.
    Returns dict with results or raises on failure.
    """
    import time as _time

    t_start = _time.time()
    cfg_path = Path(config_path).resolve()
    log_func(f"[Train] Loading config: {cfg_path}")

    is_project = cfg_path.name == "project.yaml"
    project_dir = cfg_path.parent

    if is_project:
        with open(cfg_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        with open(project_dir / "train.yaml", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}
        with open(project_dir / "dataset.yaml", encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f) or {}
        backend_name = manifest.get("backend", {}).get("name", "ultralytics")
        model_name = manifest.get("backend", {}).get("model", "yolo26n")
        task = manifest.get("task", "detection")
        project_name = manifest.get("name", project_dir.name)
    else:
        train_cfg = {}
        with open(cfg_path, encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f) or {}
        backend_name = "ultralytics"
        model_name = "yolo26n"
        task = "detection"
        project_name = cfg_path.parent.parent.name or "default"

    # Merge CLI overrides (if provided)
    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch is not None:
        train_cfg["batch"] = batch
    if imgsz is not None:
        train_cfg["imgsz"] = imgsz

    # Set model and task in config
    # 仅 ultralytics 后端需要 .pt 后缀；rfdetr 等用纯 model 名（P0-3 兼容）
    if backend_name == "ultralytics":
        train_cfg["model"] = f"{model_name}.pt" if not model_name.endswith(".pt") else model_name
    else:
        train_cfg["model"] = model_name
    train_cfg["task"] = task

    # Generate run name
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_cfg["project"] = str(project_dir / "runs")
    train_cfg["name"] = run_name

    log_func(f"  Project: {project_name}")
    log_func(f"  Backend: {backend_name}")
    log_func(f"  Model:   {model_name}")
    log_func(f"  Task:    {task}")
    log_func(f"  Device:  {device}")
    log_func(f"  Epochs:  {train_cfg.get('epochs', 'default')}")
    log_func(f"  Batch:   {train_cfg.get('batch', 'default')}")
    log_func(f"  Run:     {run_name}")

    # Parse augmentation and loss params from train.yaml
    augment = train_cfg.pop("augment", {})
    if isinstance(augment, dict):
        for k, v in augment.items():
            if k not in train_cfg:
                train_cfg[k] = v

    # Merge per-backend hyperparameters from the New Training form.
    # Reserved keys are set explicitly above / below (model/task/project/name/
    # epochs/batch/imgsz/device); each adapter's kwarg whitelist drops any
    # remaining unknown keys, so cross-backend wrong fills never reach upstream
    # (ultralytics: _filter_yolo_kwargs; rfdetr: filter_train_kwargs + warning).
    if hyperparameters:
        _reserved = {
            "model", "task", "project", "name", "data", "device",
            "epochs", "batch", "imgsz", "augment", "export",
        }
        for k, v in (hyperparameters or {}).items():
            if k in _reserved or v is None:
                continue
            train_cfg[k] = v

    # rfdetr adapter expects TrainConfig field names (batch_size / num_workers),
    # not the ultralytics-style batch / workers — normalize so the form's batch
    # and workers actually reach rf-detr training instead of being dropped.
    if backend_name == "rfdetr":
        if "batch" in train_cfg:
            # 已由 hyperparameters 提供 batch_size 时，直接丢弃 ultralytics 风格的 batch
            train_cfg.setdefault("batch_size", train_cfg.pop("batch"))
            train_cfg.pop("batch", None)
        if "workers" in train_cfg:
            train_cfg.setdefault("num_workers", train_cfg.pop("workers"))
            train_cfg.pop("workers", None)

    data_path = cfg_path if not is_project else (project_dir / "dataset.yaml")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    log_func(f"[Train] Loading adapter: {backend_name}")
    adapter_cls = load_adapter(backend_name)
    adapter = adapter_cls(train_cfg)

    log_func("[Train] Starting training...")

    try:
        result = adapter.train(
            str(data_path),
            device=device,
            log_func=log_func,
            metric_func=metric_func,
            cancel_event=cancel_event,
        )
    except KeyboardInterrupt:
        log_func("[Train] Training cancelled.")
        return {"status": "cancelled", "metrics": {}, "elapsed_seconds": round(_time.time() - t_start, 1)}

    log_func("[Train] Training complete.")
    elapsed = _time.time() - t_start
    log_func("[Train] Elapsed: {elapsed:.1f}s")

    # Extract metrics from adapter
    metrics = adapter._extract_metrics(result, "") if hasattr(adapter, "_extract_metrics") else {}

    # Also collect from results.csv
    csv_metrics = _collect_metrics_from_csv(project_dir)
    metrics.update(csv_metrics)

    # Find best model — 优先使用 adapter 返回的 model_path（rf-detr 等非 ultralytics 布局）
    best_pt = None
    if isinstance(result, dict):
        mp = result.get("model_path")
        if mp and Path(mp).exists():
            best_pt = Path(mp)
    if best_pt is None:
        best_pt = _find_best_model(project_dir, run_name)

    # Save to models directory

    models_dir = R.models_dir(project_name)
    models_dir.mkdir(parents=True, exist_ok=True)
    # P2-16: 版本号递增而非硬编码 v1.0
    from ._models import next_model_path
    dst = next_model_path(models_dir)
    # P1-6: 后缀按实际 checkpoint 确定（rf-detr .pth / yolo .pt）
    if best_pt and best_pt.exists():
        dst = dst.with_suffix(best_pt.suffix)
    model_version = dst.stem  # v1.0 / v1.1 ...
    model_version_meta = model_version.lstrip("v")  # metadata 用纯版本号
    saved_model = None

    if best_pt and best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(dst))
        saved_model = str(dst)
        log_func(f"[Train] Model saved: {dst}")

        meta_path = dst.with_suffix(".metadata.yaml")
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "model": model_name,
                    "version": model_version,
                    "project": project_name,
                    "created_at": datetime.now().isoformat(),
                    "metrics": metrics,
                },
                f,
                default_flow_style=False,
            )

    # Export ONNX with optional verification
    onnx_path = None
    onnx_verification = None
    if best_pt and best_pt.exists():
        try:
            # Look for dataset.yaml for verification
            ds_yaml = str(data_path)
            onnx_path = str(models_dir / f"{model_version}.onnx")

            export_result = adapter.export(
                onnx_path,
                fmt=export_format,
                device=device,
                model_path=str(best_pt),
                opset=export_opset,
                half=export_half,
                int8=export_int8,
                dynamic=export_dynamic,
                simplify=export_simplify,
                dataset_yaml=ds_yaml if export_format == "onnx" else None,
                verify=(export_format == "onnx"),
                metadata={"model_name": model_name, "model_version": model_version_meta},
                log_func=log_func,
            )
            if export_result.get("onnx_path"):
                onnx_path = export_result["onnx_path"]
                onnx_verification = export_result.get("verification")
                log_func(f"[Train] ONNX exported: {onnx_path}")
                if onnx_verification:
                    v_status = onnx_verification.get("status", "unknown")
                    log_func(f"[Train] ONNX verification: {v_status}")
        except Exception as e:
            log_func(f"[Train] ONNX export skipped: {e}")

    # P2: 训练完成后自动注册 ONNX 模型到 raven-go 推理平台
    platform_registered = False
    if register_to_platform and onnx_path:
        try:
            from ..integration.raven_go import register_model_to_platform
            log_func("[Train] Registering model to raven-go platform...")
            reg_result = register_model_to_platform(
                onnx_path=onnx_path,
                model_name=model_name,
                task=task,
                project_name=project_name,
                metrics=metrics,
                quantization={"half": export_half, "int8": export_int8},
                log_func=log_func,
            )
            platform_registered = reg_result.get("success", False)
            if platform_registered:
                log_func(f"[Train] Model registered to platform: {reg_result.get('model_id', 'unknown')}")
            else:
                log_func(f"[Train] Platform registration failed: {reg_result.get('error', 'unknown')}")
        except Exception as e:
            log_func(f"[Train] Platform registration skipped: {e}")

    # Collect reproducibility metadata
    reproducibility: Dict[str, Any] = {}
    try:
        import subprocess as _sp
        r = _sp.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            reproducibility["git_commit"] = r.stdout.strip()
    except Exception:
        pass
    reproducibility["adapter"] = backend_name
    try:
        import ultralytics
        reproducibility["ultralytics_version"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import torch
        reproducibility["torch_version"] = torch.__version__
        reproducibility["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass

    # Build dataset snapshot
    dataset_snapshot = {
        "config": str(data_path),
        "name": is_project and manifest.get("name", "") or data_path.parent.name,
    }
    try:
        if data_path.exists():
            ds_cfg = yaml.safe_load(open(data_path, encoding="utf-8"))
            dataset_snapshot["train"] = ds_cfg.get("train", "")
            dataset_snapshot["val"] = ds_cfg.get("val", "")
            dataset_snapshot["nc"] = ds_cfg.get("nc", 0)
    except Exception:
        pass

    result_dict = {
        "status": "completed",
        "metrics": metrics,
        "best_model": saved_model,
        "onnx_path": onnx_path,
        "onnx_verification": onnx_verification,
        "platform_registered": platform_registered,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save experiment
    exp_path = _save_experiment(
        project_name=project_name,
        config={
            "backend": backend_name,
            "model": model_name,
            "task": task,
            "device": device,
            "epochs": train_cfg.get("epochs"),
            "batch": train_cfg.get("batch"),
            "imgsz": train_cfg.get("imgsz"),
            "dataset": dataset_snapshot,
            "reproducibility": reproducibility,
        },
        results=result_dict,
        model_path=Path(saved_model) if saved_model else None,
    )
    log_func(f"[Train] Experiment: {exp_path}")

    # Generate training report
    try:
        from ..report import generate_training_report
        report_path = project_dir / "runs" / run_name / "train_report.html"
        if not report_path.parent.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_training_report(
            exp_path=exp_path,
            project_name=project_name,
            config={
                "backend": backend_name,
                "model": model_name,
                "task": task,
                "epochs": train_cfg.get("epochs"),
                "batch": train_cfg.get("batch"),
                "imgsz": train_cfg.get("imgsz"),
                "dataset": dataset_snapshot,
            },
            metrics=metrics,
            model_path=saved_model,
            onnx_path=Path(onnx_path) if onnx_path and Path(onnx_path).exists() else None,
            output_path=report_path,
        )
        log_func(f"[Train] Report: {report_path}")
    except Exception as e:
        log_func(f"[Train] Report generation skipped: {e}")

    log_func("[Train] Done.")
    return result_dict


@click.command(name="train")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--device", default="cuda:0", help="Training device")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--epochs", type=int, default=None, help="Override epochs")
@click.option("--batch", type=int, default=None, help="Override batch size")
@click.option("--imgsz", type=int, default=None, help="Override image size")
@click.option("--name", default=None, help="Experiment name suffix")
@click.option("--half", is_flag=True, help="FP16 quantization on ONNX export")
@click.option("--int8", is_flag=True, help="INT8 quantization on ONNX export (requires dataset.yaml)")
@click.option("--register", "register_to_platform", is_flag=True,
              help="Register trained ONNX model to raven-go platform after training")
def cmd_train(
    config_path: str,
    device: str,
    seed: int,
    resume: bool,
    epochs: Optional[int],
    batch: Optional[int],
    imgsz: Optional[int],
    name: Optional[str],
    half: bool,
    int8: bool,
    register_to_platform: bool,
):
    """Train a model.

    CONFIG_PATH can be:
    - A project.yaml (from 'raven-train new')
    - A dataset.yaml (from 'dataset split')
    """
    jm = JobManager()
    job = jm.create(
        job_type=JobType.TRAIN,
        config={"config_path": config_path, "device": device, "epochs": epochs, "batch": batch, "imgsz": imgsz},
        label=f"Train {config_path}",
    )
    jm.update_status(job.job_id, JobStatus.RUNNING)
    jm.append_log(job.job_id, f"Starting training: {config_path}")
    t_start = time.time()
    cfg_path = Path(config_path).resolve()
    click.echo(f"[Train] Loading config: {cfg_path}")

    is_project = cfg_path.name == "project.yaml"
    project_dir = cfg_path.parent

    if is_project:
        manifest = _load_yaml(cfg_path)
        train_cfg = _load_yaml(project_dir / "train.yaml")
        dataset_cfg = _load_yaml(project_dir / "dataset.yaml")
        backend_name = manifest.get("backend", {}).get("name", "ultralytics")
        model_name = manifest.get("backend", {}).get("model", "yolo26n")
        task = manifest.get("task", "detection")
        project_name = manifest.get("name", project_dir.name)
    else:
        train_cfg = {}
        dataset_cfg = _load_yaml(cfg_path)
        backend_name = "ultralytics"
        model_name = "yolo26n"
        task = "detection"
        project_name = cfg_path.parent.parent.name or "default"

    # Merge CLI overrides
    if epochs:
        train_cfg["epochs"] = epochs
    if batch:
        train_cfg["batch"] = batch
    if imgsz:
        train_cfg["imgsz"] = imgsz

    # Set model and task
    # 仅 ultralytics 后端需要 .pt 后缀；rfdetr 等用纯 model 名（P0-3 兼容）
    if backend_name == "ultralytics":
        pt_model = f"{model_name}.pt" if not model_name.endswith(".pt") else model_name
        train_cfg["model"] = pt_model
    else:
        train_cfg["model"] = model_name
    train_cfg["task"] = task

    # Parse augmentation params from train.yaml's 'augment' section
    augment = train_cfg.pop("augment", {})
    if isinstance(augment, dict):
        for k, v in augment.items():
            if k not in train_cfg:
                train_cfg[k] = v

    # Generate run name
    exp_suffix = f"_{name}" if name else ""
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{exp_suffix}"

    click.echo(f"\n  Project:    {project_name}")
    click.echo(f"  Backend:    {backend_name}")
    click.echo(f"  Model:      {model_name}")
    click.echo(f"  Task:       {task}")
    click.echo(f"  Device:     {device}")
    click.echo(f"  Epochs:     {train_cfg.get('epochs', 'default')}")
    click.echo(f"  Batch:      {train_cfg.get('batch', 'default')}")
    click.echo(f"  Image size: {train_cfg.get('imgsz', 'default')}")
    click.echo(f"  Run name:   {run_name}")

    # Set project/name in config
    train_cfg["project"] = str(project_dir / "runs")
    train_cfg["name"] = run_name

    # Load adapter
    click.echo(f"\n[Train] Loading adapter: {backend_name}")
    adapter_cls = load_adapter(backend_name)
    adapter = adapter_cls(train_cfg)

    # Determine data config path
    data_path = cfg_path if not is_project else (project_dir / "dataset.yaml")
    if not data_path.exists():
        click.echo(f"[Train] Dataset config not found: {data_path}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=f"Dataset config not found: {data_path}")
        return

    # Build extra training kwargs from CLI flags
    train_extra_kwargs: dict = {}
    if resume:
        train_extra_kwargs["resume"] = True
        click.echo("[Train] Resume from last checkpoint: enabled")

    click.echo("[Train] Starting training...")
    try:
        result = adapter.train(
            str(data_path),
            device=device,
            seed=seed,
            **train_extra_kwargs,
        )
        click.echo("[Train] Training complete.")
    except KeyboardInterrupt:
        click.echo("[Train] Training cancelled by user.")
        jm.update_status(job.job_id, JobStatus.CANCELLED, error="Cancelled by user")
        return
    except Exception as e:
        click.echo(f"[Train] Training FAILED: {e}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=str(e))
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t_start
    click.echo("[Train] Elapsed: {elapsed:.1f}s")

    # Extract metrics
    metrics = adapter._extract_metrics(result, "") if hasattr(adapter, "_extract_metrics") else {}
    csv_metrics = _collect_metrics_from_csv(project_dir)
    metrics.update(csv_metrics)

    if metrics:
        click.echo(f"[Train] Metrics: {json.dumps(metrics, indent=2)}")

    # Find best model — 优先使用 adapter 返回的 model_path（rf-detr 等非 ultralytics 布局）
    best_pt = None
    if isinstance(result, dict):
        mp = result.get("model_path")
        if mp and Path(mp).exists():
            best_pt = Path(mp)
    if best_pt is None:
        best_pt = _find_best_model(project_dir, run_name)
    last_pt = project_dir / "runs" / run_name / "weights" / "last.pt"

    # Save to models directory
    models_dir = R.models_dir(project_name)
    models_dir.mkdir(parents=True, exist_ok=True)

    # P2-16: 版本号递增而非硬编码 v1.0
    from ._models import next_model_path
    dst = next_model_path(models_dir)
    # P1-6: 后缀按实际 checkpoint 确定（rf-detr .pth / yolo .pt）
    if best_pt and best_pt.exists():
        dst = dst.with_suffix(best_pt.suffix)
    model_version = dst.stem  # v1.0 / v1.1 ...
    model_version_meta = model_version.lstrip("v")  # metadata 用纯版本号
    saved_model = None
    if best_pt and best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(dst))
        saved_model = dst
        click.echo(f"[Train] Model saved: {dst}")

        # Save model metadata
        meta_path = dst.with_suffix(".metadata.yaml")
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "model": model_name,
                    "version": model_version,
                    "project": project_name,
                    "created_at": datetime.now().isoformat(),
                    "metrics": metrics,
                },
                f,
                default_flow_style=False,
            )
        click.echo(f"[Train] Model metadata: {meta_path}")

    # Build dataset snapshot
    dataset_snapshot = {
        "config": str(data_path),
        "name": is_project and manifest.get("name", "") or data_path.parent.name,
    }
    try:
        if data_path.exists():
            ds_cfg = yaml.safe_load(open(data_path, encoding="utf-8"))
            dataset_snapshot["train"] = ds_cfg.get("train", "")
            dataset_snapshot["val"] = ds_cfg.get("val", "")
            dataset_snapshot["nc"] = ds_cfg.get("nc", 0)
    except Exception:
        pass

    # Collect reproducibility metadata
    reproducibility: Dict[str, Any] = {}
    try:
        import subprocess as _sp
        r = _sp.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            reproducibility["git_commit"] = r.stdout.strip()
    except Exception:
        pass
    reproducibility["adapter"] = backend_name
    try:
        import ultralytics
        reproducibility["ultralytics_version"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import torch
        reproducibility["torch_version"] = torch.__version__
        reproducibility["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass

    # Save experiment
    exp_path = _save_experiment(
        project_name=project_name,
        config={
            "backend": backend_name,
            "model": model_name,
            "task": task,
            "device": device,
            "seed": seed,
            "epochs": train_cfg.get("epochs"),
            "batch": train_cfg.get("batch"),
            "imgsz": train_cfg.get("imgsz"),
            "dataset": dataset_snapshot,
            "reproducibility": reproducibility,
        },
        results={"elapsed_seconds": round(elapsed, 1), **metrics},
        model_path=saved_model,
    )
    click.echo(f"[Train] Experiment: {exp_path}")

    # Export ONNX automatically with verification
    onnx_exported_path = None
    if best_pt and best_pt.exists():
        try:
            click.echo("[Train] Exporting ONNX...")
            onnx_path = str(models_dir / f"{model_version}.onnx")
            ds_yaml = str(data_path)
            export_result = adapter.export(
                onnx_path,
                fmt="onnx",
                device=device,
                model_path=str(best_pt),
                opset=17,
                half=half,
                int8=int8,
                # P0-2: 与 EXPORT_SPEC（动态 batch=Yes）对齐
                dynamic=True,
                simplify=True,
                dataset_yaml=ds_yaml,
                verify=True,
                metadata={"model_name": model_name, "model_version": model_version_meta},
                log_func=lambda msg: click.echo(msg),
            )
            if export_result.get("onnx_path"):
                onnx_exported_path = export_result["onnx_path"]
                click.echo(f"[Train] ONNX exported: {onnx_exported_path}")
                if "verification" in export_result:
                    v = export_result["verification"]
                    v_status = v.get("status", "unknown")
                    click.echo(f"[Train] ONNX verification: {v_status}")
                    if "onnx_val" in v:
                        val_info = v["onnx_val"]
                        click.echo(f"  mAP50: {val_info.get('map50', 'N/A')}")
                        click.echo(f"  mAP50-95: {val_info.get('map50_95', 'N/A')}")
        except Exception as e:
            click.echo(f"[Train] ONNX export skipped: {e}")

    # P2: 训练完成后自动注册 ONNX 模型到 raven-go 推理平台
    if register_to_platform and onnx_exported_path:
        try:
            from ..integration.raven_go import register_model_to_platform
            click.echo("[Train] Registering model to raven-go platform...")
            reg_result = register_model_to_platform(
                onnx_path=onnx_exported_path,
                model_name=model_name,
                task=task,
                project_name=project_name,
                metrics=metrics,
                quantization={"half": half, "int8": int8},
                log_func=lambda msg: click.echo(msg),
            )
            if reg_result.get("success"):
                click.echo(f"[Train] Model registered to platform: {reg_result.get('model_id', 'unknown')}")
            else:
                click.echo(f"[Train] Platform registration failed: {reg_result.get('error', 'unknown')}")
        except Exception as e:
            click.echo(f"[Train] Platform registration skipped: {e}")

    # Generate training report
    try:
        from ..report import generate_training_report
        report_path = project_dir / "runs" / run_name / "train_report.html"
        if not report_path.parent.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_training_report(
            exp_path=exp_path,
            project_name=project_name,
            config={
                "backend": backend_name,
                "model": model_name,
                "task": task,
                "epochs": train_cfg.get("epochs"),
                "batch": train_cfg.get("batch"),
                "imgsz": train_cfg.get("imgsz"),
                "dataset": dataset_snapshot,
            },
            metrics=metrics,
            model_path=saved_model,
            onnx_path=Path(str(models_dir / f"{model_version}.onnx")) if saved_model else None,
            output_path=report_path,
        )
        click.echo(f"[Train] Report: {report_path}")
    except Exception as e:
        click.echo(f"[Train] Report generation skipped: {e}")

    jm.save_result(job.job_id, {
        "project_name": project_name,
        "model": model_name,
        "best_model": str(saved_model) if saved_model else None,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "experiment": str(exp_path),
    })
    jm.append_log(job.job_id, f"Training complete. Model: {saved_model}")
    jm.update_status(job.job_id, JobStatus.COMPLETED)
    click.echo(f"\n[Train] Done. Use 'raven-train export {cfg_path}' to re-export.")
