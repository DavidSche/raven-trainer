"""Training API — 训练管理。

支持完整的 ultralytics 超参数透传、实时日志 WebSocket、
训练取消、ONNX 约束一致性验证。
"""

from __future__ import annotations

import asyncio
import json
import threading
from ...data_asset.job import JobManager, JobStatus, JobType
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..schemas import ExperimentSchema, ExperimentConfigSchema
from ...config import R
from ...experiments import ExperimentManager, scan_all_experiments, Experiment

router = APIRouter(tags=["training"])


# ── Active training tracking ──

_active_training: Dict[str, Dict[str, Any]] = {}
_ws_clients: Dict[str, List[WebSocket]] = {}
_cancel_events: Dict[str, threading.Event] = {}
# Reference to the main asyncio event loop, captured at app startup so that
# background training threads can schedule WebSocket sends via
# asyncio.run_coroutine_threadsafe(coro, loop). Python 3.12+ no longer
# returns a usable loop from get_event_loop() in non-main threads.
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture the main event loop — called from server startup."""
    global _main_loop
    _main_loop = loop


def _set_active_status(exp_id: str, status: str) -> None:
    """Update the in-memory status of an active training run.

    Tolerates the entry having already been removed (e.g. the experiment was
    deleted from the UI while the background thread was still finishing).
    """
    entry = _active_training.get(exp_id)
    if entry is not None:
        entry["status"] = status


def _exp_to_schema(exp: Experiment) -> ExperimentSchema:
    return ExperimentSchema(
        id=exp.experiment_id,
        project=exp.project,
        model=exp.model,
        backend=exp.backend,
        status=exp.status,
        epochs=exp.epochs,
        batch=exp.batch,
        imgsz=exp.imgsz,
        device=exp.device,
        metrics=exp.metrics,
        config=exp.config,
        created_at=exp.created_at,
        finished_at=exp.finished_at,
        dataset=exp.dataset,
        workspace=exp.workspace,
    )


# ── List / Detail ──


@router.get("/training/experiments")
def list_experiments(limit: int = Query(50)):
    experiments = scan_all_experiments()
    experiments.sort(key=lambda x: x.created_at, reverse=True)
    return [_exp_to_schema(e) for e in experiments[:limit]]


@router.get("/training/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            return _exp_to_schema(e)
    raise HTTPException(404, f"Experiment '{experiment_id}' not found")


@router.get("/training/experiments/{experiment_id}/config")
def get_experiment_config(experiment_id: str):
    """Get full experiment configuration with reproducibility info."""
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            return {
                "experiment_id": e.experiment_id,
                "config": e.config,
                "reproducibility": e.reproducibility,
            }
    raise HTTPException(404, f"Experiment '{experiment_id}' not found")


@router.delete("/training/experiments/{experiment_id}")
def delete_experiment(experiment_id: str):
    """Delete an experiment record and its associated files."""
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            ExperimentManager(e.project).delete(experiment_id)
            _active_training.pop(experiment_id, None)
            _ws_clients.pop(experiment_id, None)
            _cancel_events.pop(experiment_id, None)
            return {"status": "deleted", "experiment_id": experiment_id}
    raise HTTPException(404, f"Experiment '{experiment_id}' not found")


@router.get("/training/experiments/{experiment_id}/log")
def get_experiment_log(experiment_id: str):
    """Get training log content."""
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            mgr = ExperimentManager(e.project)
            log_content = mgr.get_log(experiment_id)
            return {"experiment_id": experiment_id, "log": log_content}
    raise HTTPException(404, f"Experiment '{experiment_id}' not found")


@router.get("/training/experiments/{experiment_id}/report")
def get_experiment_report(experiment_id: str):
    """Get training report HTML content."""
    # Only serve reports for experiments that actually exist in the registry;
    # this keeps the path fully derived from trusted metadata.
    known = next((e for e in scan_all_experiments() if e.experiment_id == experiment_id), None)
    if known is None:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    for scan_root in [R.PROJECTS]:
        if not scan_root.exists():
            continue
        for proj_dir in scan_root.iterdir():
            if not proj_dir.is_dir():
                continue
            report_path = proj_dir / "experiments" / experiment_id / "train_report.html"
            if report_path.exists():
                return {"html": report_path.read_text(encoding="utf-8")}
            report_path = proj_dir / "reports" / f"{experiment_id}.html"
            if report_path.exists():
                return {"html": report_path.read_text(encoding="utf-8")}
    raise HTTPException(404, f"Report for experiment '{experiment_id}' not found")


@router.get("/training/experiments/{experiment_id}/results")
def get_experiment_results(experiment_id: str):
    """Get training results: CSV data and list of generated images.

    Ported from aiva-trainer: parses results.csv, lists plot images.
    """
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            return _collect_experiment_results(e)
    raise HTTPException(404, f"Experiment '{experiment_id}' not found")


def _collect_experiment_results(exp: Experiment) -> Dict[str, Any]:
    """Collect training results: CSV, images, args.yaml."""
    project_dir = R.project_dir(exp.project)
    run_dir = None

    # Try to find run directory
    best_model = exp.best_model_path
    if best_model:
        p = Path(best_model).parent.parent
        if p.exists():
            run_dir = p

    if not run_dir:
        exp_dirs = sorted((project_dir / "runs").iterdir()) if (project_dir / "runs").exists() else []
        for d in reversed(exp_dirs):
            if d.is_dir():
                run_dir = d
                break

    if not run_dir or not run_dir.exists():
        return {"exists": False, "message": "Run directory not found"}

    # Read results.csv
    csv_data = []
    headers: List[str] = []
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            headers = [h.strip() for h in lines[0].split(",")]
            for line in lines[1:]:
                line = line.strip()
                if line:
                    values = line.split(",")
                    row: Dict[str, Any] = {}
                    for i, h in enumerate(headers):
                        if i < len(values):
                            try:
                                row[h] = float(values[i])
                            except ValueError:
                                row[h] = values[i]
                    csv_data.append(row)

    # List image files
    image_extensions = {".png", ".jpg", ".jpeg"}
    images = []
    for img_path in sorted(run_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            images.append({
                "name": img_path.name,
                "path": str(img_path.resolve()),
                "size": img_path.stat().st_size,
            })

    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        for img_path in sorted(weights_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                images.append({
                    "name": f"weights/{img_path.name}",
                    "path": str(img_path.resolve()),
                    "size": img_path.stat().st_size,
                })

    args_content = ""
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        args_content = args_path.read_text(encoding="utf-8")

    return {
        "exists": True,
        "run_dir": str(run_dir.resolve()),
        "csv": {
            "headers": headers,
            "rows": csv_data,
        },
        "images": images,
        "args_yaml": args_content,
    }


# ── Start Training ──


@router.post("/training/start")
def start_training(cfg: ExperimentConfigSchema):
    """Start training in a background thread with full hyperparameter support."""
    project_dir = R.project_dir(cfg.project)
    if not project_dir.exists():
        raise HTTPException(
            400, f"Project '{cfg.project}' not found. Create it with `raven-train new`"
        )

    jm = JobManager()
    job = jm.create(
        job_type=JobType.TRAIN,
        config={"project": cfg.project, "model": cfg.model, "backend": cfg.backend},
        label=f"Train {cfg.project}/{cfg.model}",
    )
    jm.update_status(job.job_id, JobStatus.RUNNING)
    jm.append_log(job.job_id, f"Starting training: {cfg.project}/{cfg.model}")
    mgr = ExperimentManager(cfg.project)

    # Load full training hyperparams from train.yaml
    train_yaml_path = project_dir / "train.yaml"
    train_hyperparams: Dict[str, Any] = {}
    if train_yaml_path.exists():
        import yaml as _yaml
        try:
            with open(train_yaml_path, encoding="utf-8") as _f:
                train_hyperparams = _yaml.safe_load(_f) or {}
        except Exception:
            pass

    # Parse augment section
    augment = train_hyperparams.pop("augment", {})
    if isinstance(augment, dict):
        for k, v in augment.items():
            if k not in train_hyperparams:
                train_hyperparams[k] = v

    exp = mgr.create(
        model=cfg.model,
        backend=cfg.backend,
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        dataset=cfg.dataset,
        workspace=cfg.workspace,
        device=cfg.device,
        status="pending",
        config={
            "model": cfg.model,
            "backend": cfg.backend,
            "epochs": cfg.epochs,
            "batch": cfg.batch,
            "imgsz": cfg.imgsz,
            "device": cfg.device,
            "dataset": cfg.dataset,
            "workspace": cfg.workspace,
            "hyperparameters": train_hyperparams,
            "form_hyperparameters": cfg.hyperparameters,
        },
    )
    mgr.update_status(exp.experiment_id, "running")

    # Create cancel event for this experiment
    cancel_event = threading.Event()
    _cancel_events[exp.experiment_id] = cancel_event

    # Start background training
    thread = threading.Thread(
        target=_run_training_thread,
        args=(cfg, exp, mgr, cancel_event, job.job_id),
        daemon=True,
    )
    thread.start()

    return {
        "experiment_id": exp.experiment_id,
        "status": "running",
        "message": f"Training started. Use WS /api/v1/ws/training/{exp.experiment_id} for realtime logs.",
    }


def _finish_job(job_id: str, status: JobStatus, error: str = "") -> None:
    """Close out the job record created by ``start_training``.

    The job exists so the UI can list the run as an active operation; without
    this the record would stay ``running`` forever (and be marked failed by the
    startup recovery sweep after a restart).
    """
    if not job_id:
        return
    try:
        JobManager().update_status(job_id, status, error=error)
    except (ValueError, OSError):
        pass


def _run_training_thread(
    cfg: ExperimentConfigSchema,
    exp: Experiment,
    mgr: ExperimentManager,
    cancel_event: threading.Event,
    job_id: str = "",
):
    """Run training in background thread with log streaming and cancellation."""
    exp_id = exp.experiment_id
    log_lines: List[str] = []
    job_status = JobStatus.FAILED
    job_error = ""

    def _log(msg: str):
        log_lines.append(msg)
        mgr.append_log(exp_id, msg)
        _broadcast_log(exp_id, msg)

    # P2: per-epoch 结构化指标回调，实时推送 loss/mAP 到前端 WS
    def _metric_func(metric_data: dict):
        _broadcast_metric(exp_id, metric_data)

    _active_training[exp_id] = {"status": "running", "log": log_lines}

    try:
        project_dir = R.project_dir(cfg.project)
        config_path = project_dir / "project.yaml"

        _log(f"[Train] Starting {exp_id}")
        _log(f"  Project: {cfg.project}")
        _log(f"  Model:   {cfg.model}")
        _log(f"  Backend: {cfg.backend}")
        _log(f"  Epochs:  {cfg.epochs}")
        _log(f"  Batch:   {cfg.batch}")
        _log(f"  Device:  {cfg.device}")
        _log("")

        _log("[Train] Calling training function...")

        from ...cli.train import _do_train

        try:
            result = _do_train(
                config_path=config_path,
                device=cfg.device,
                epochs=cfg.epochs,
                batch=cfg.batch,
                imgsz=cfg.imgsz,
                log_func=_log,
                metric_func=_metric_func,
                export_format=cfg.export_format,
                export_opset=cfg.export_opset,
                export_half=cfg.export_half,
                export_int8=cfg.export_int8,
                export_dynamic=cfg.export_dynamic,
                export_simplify=cfg.export_simplify,
                cancel_event=cancel_event,
                register_to_platform=cfg.register_to_platform,
                hyperparameters=cfg.hyperparameters,
            )

            if result and result.get("status") == "cancelled":
                mgr.update_status(exp_id, "cancelled")
                _log(f"[Train] {exp_id} cancelled.")
                _active_training[exp_id]["status"] = "cancelled"
                job_status = JobStatus.CANCELLED
            elif cancel_event.is_set():
                # The user pressed stop and the adapter returned normally anyway:
                # never rewrite a user-cancelled run as completed.
                mgr.update_status(exp_id, "cancelled")
                _log(f"[Train] {exp_id} cancelled.")
                _set_active_status(exp_id, "cancelled")
                job_status = JobStatus.CANCELLED
            else:
                mgr.update_status(exp_id, "completed")
                _log(f"[Train] {exp_id} completed successfully")
                _active_training[exp_id]["status"] = "completed"
                job_status = JobStatus.COMPLETED

                # Collect metrics
                if result:
                    if "metrics" in result:
                        mgr.update_metrics(exp_id, result["metrics"])
                        _log(f"  Metrics: {json.dumps(result['metrics'], indent=2)}")
                    if "best_model" in result:
                        _log(f"  Model: {result['best_model']}")
                    if "onnx_verification" in result and result["onnx_verification"]:
                        v = result["onnx_verification"]
                        v_status = v.get("status", "unknown")
                        _log(f"  ONNX verification: {v_status}")
                        if "onnx_val" in v:
                            _log(f"    mAP50: {v['onnx_val'].get('map50', 'N/A')}")
                            _log(f"    mAP50-95: {v['onnx_val'].get('map_95', 'N/A')}")
                    if result.get("platform_registered"):
                        _log("  Platform: model registered to raven-go")

                # Generate training report
                try:
                    from ...report import generate_training_report

                    exp_dir = R.experiment_dir(cfg.project) / exp_id
                    exp_report_path = exp_dir / "train_report.html"

                    import yaml as _yaml

                    ds_yaml = project_dir / "dataset.yaml"
                    dataset_cfg: Dict[str, Any] = {}
                    if ds_yaml.exists():
                        with open(ds_yaml, encoding="utf-8") as _f:
                            dataset_cfg = _yaml.safe_load(_f) or {}

                    cfg_dict = {
                        "backend": cfg.backend,
                        "model": cfg.model,
                        "task": "detection",
                        "epochs": cfg.epochs,
                        "batch": cfg.batch,
                        "imgsz": cfg.imgsz,
                        "dataset": {
                            "name": dataset_cfg.get("names", {}).get("0", cfg.dataset or ""),
                            "train": dataset_cfg.get("train", ""),
                            "val": dataset_cfg.get("val", ""),
                            "nc": dataset_cfg.get("nc", 0),
                        },
                    }
                    metrics_dict = (result or {}).get("metrics", {})
                    model_path_str = (result or {}).get("best_model", "")
                    onnx_path_str = (result or {}).get("onnx_path", "")
                    onnx_path_obj = (
                        Path(onnx_path_str) if onnx_path_str and Path(onnx_path_str).exists() else None
                    )

                    generate_training_report(
                        exp_path=str(exp_dir / "experiment.yaml"),
                        project_name=cfg.project,
                        config=cfg_dict,
                        metrics=metrics_dict,
                        model_path=model_path_str,
                        onnx_path=onnx_path_obj,
                        output_path=exp_report_path,
                    )
                    _log(f"[Train] Report generated: {exp_report_path}")
                except Exception as report_e:
                    _log(f"[Train] Report generation: {report_e}")

        except Exception as e:
            mgr.update_status(exp_id, "failed", error=str(e))
            _log(f"[Train] {exp_id} FAILED: {e}")
            _set_active_status(exp_id, "failed")
            job_error = str(e)
            import traceback
            _log(traceback.format_exc())

    except Exception as e:
        mgr.update_status(exp_id, "failed", error=str(e))
        _log(f"[Train] ERROR: {e}")
        _set_active_status(exp_id, "failed")
        job_error = str(e)
        import traceback
        _log(traceback.format_exc())
    finally:
        # The cancel event is only useful while the worker is alive.
        _cancel_events.pop(exp_id, None)
        _finish_job(job_id, job_status, job_error)


# ── Stop / Status ──


@router.post("/training/stop")
def stop_training(data: dict):
    """Stop a running training by setting cancellation event."""
    experiment_id = data.get("experiment_id", "")
    if not experiment_id:
        raise HTTPException(400, "experiment_id required")

    # Set cancellation event — the adapter checks this on each epoch end
    cancel_event = _cancel_events.get(experiment_id)
    if cancel_event:
        cancel_event.set()

    # Update status only while the experiment is still running; a late "stop"
    # click must not rewrite a completed/failed record as cancelled.
    experiments = scan_all_experiments()
    for e in experiments:
        if e.experiment_id == experiment_id:
            if e.status not in ("running", "pending"):
                return {
                    "status": "not-running",
                    "experiment_id": experiment_id,
                    "current_status": e.status,
                }
            mgr = ExperimentManager(e.project)
            mgr.update_status(experiment_id, "cancelled")
            break

    _broadcast_log(experiment_id, "[Train] Stop requested by user.")
    return {"status": "stopped", "experiment_id": experiment_id}


@router.get("/training/status")
def training_status():
    """Check status of all training runs."""
    results = {}
    for eid, info in _active_training.items():
        log_preview = info.get("log", [])[-10:] if info.get("log") else []
        results[eid] = {
            "status": info.get("status", "unknown"),
            "log_preview": log_preview,
        }
    return results


# ── Experiment Compare ──


@router.post("/training/experiments/compare")
def compare_experiments(data: dict):
    """Compare multiple experiments."""
    ids = data.get("experiment_ids", [])
    if not ids:
        raise HTTPException(400, "experiment_ids required")

    experiments = scan_all_experiments()
    selected = [e for e in experiments if e.experiment_id in ids]

    if not selected:
        raise HTTPException(404, "No experiments found")

    return {
        "experiments": [_exp_to_schema(e).model_dump() for e in selected],
        "comparison": {
            "total": len(selected),
            "fields": [
                {"key": "model", "label": "Model"},
                {"key": "epochs", "label": "Epochs"},
                {"key": "batch", "label": "Batch"},
                {"key": "imgsz", "label": "Image Size"},
                {"key": "status", "label": "Status"},
            ],
            "metrics": list(
                dict.fromkeys(k for e in selected for k in e.metrics.keys())
            ),
        },
    }




# ── Training Project Management ──

from pydantic import BaseModel


class CreateProjectRequest(BaseModel):
    name: str
    task: str = "detection"
    backend: str = "ultralytics"
    model: str = "yolo26n"
    dataset_path: str = ""
    dataset_yaml: str = ""


@router.post("/training/projects")
def create_training_project(req: CreateProjectRequest):
    """Create a new training project (equivalent to `raven-train new`)."""
    from ...cli.new import ProjectScaffolder

    # Determine dataset source from path
    ds_source = "local"
    if "cvat" in req.dataset_path.lower():
        ds_source = "cvat"
    elif "label" in req.dataset_path.lower() or "studio" in req.dataset_path.lower():
        ds_source = "label_studio"

    scaffolder = ProjectScaffolder(
        name=req.name,
        task=req.task,
        backend=req.backend,
        model=req.model,
        dataset_source=ds_source,
    )

    try:
        project_dir = scaffolder.run()
    except FileExistsError as exc:
        raise HTTPException(400, f"Project '{req.name}' already exists") from exc

    # Update dataset.yaml with actual paths if provided
    if req.dataset_yaml:
        # A client-supplied YAML path must stay inside the data roots: this
        # endpoint copies the file into the new project, so an unchecked path
        # would let a caller read any file the server can read.
        import yaml
        ds_src = Path(req.dataset_yaml).expanduser()
        if not ds_src.is_absolute():
            ds_src = R.PROJECT_ROOT / ds_src
        try:
            ds_src = ds_src.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(404, f"dataset_yaml not found: {req.dataset_yaml}") from exc
        allowed_roots = (R.PROJECT_ROOT.resolve(), R.DATA.resolve())
        if not any(ds_src.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(400, "dataset_yaml must point inside the project or data directory")
        with open(ds_src, encoding="utf-8") as f:
            real_ds = yaml.safe_load(f) or {}
        with open(project_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(real_ds, f, default_flow_style=False, sort_keys=False)

    return {
        "status": "created",
        "name": req.name,
        "project_dir": str(project_dir),
        "project_yaml": str(project_dir / "project.yaml"),
        "dataset_yaml": str(project_dir / "dataset.yaml"),
    }


# ── WebSocket for real-time logs ──


@router.websocket("/ws/training/{experiment_id}")
async def training_websocket(websocket: WebSocket, experiment_id: str):
    await websocket.accept()
    if experiment_id not in _ws_clients:
        _ws_clients[experiment_id] = []
    _ws_clients[experiment_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    except Exception:
        # Any other exception (client crash, network error) must still clean up
        # to avoid leaking sockets in _ws_clients
        pass
    finally:
        if experiment_id in _ws_clients:
            try:
                _ws_clients[experiment_id].remove(websocket)
            except ValueError:
                pass
            if not _ws_clients[experiment_id]:
                del _ws_clients[experiment_id]


# ── Training result image serving ──


@router.get("/training/experiments/{experiment_id}/results-image")
def get_experiment_result_image(experiment_id: str, name: str = ""):
    """Serve a training result image file.

    Ported from aiva-trainer: serves plot images from run directory.
    """
    from fastapi.responses import FileResponse

    experiments = scan_all_experiments()
    exp = None
    for e in experiments:
        if e.experiment_id == experiment_id:
            exp = e
            break

    if not exp:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")

    # Find run directory
    exp_dir = None
    if exp.best_model_path:
        p = Path(exp.best_model_path).parent.parent
        if p.exists():
            exp_dir = p
    if not exp_dir:
        project_dir = R.project_dir(exp.project)
        p = project_dir / "runs" / experiment_id
        if p.exists():
            exp_dir = p
    if not exp_dir:
        project_dir = R.project_dir(exp.project)
        run_dirs = sorted((project_dir / "runs").iterdir()) if (project_dir / "runs").exists() else []
        for d in reversed(run_dirs):
            if d.is_dir():
                exp_dir = d
                break

    if not exp_dir:
        raise HTTPException(404, "Run directory not found")

    if ".." in name or name.startswith("/") or name.startswith("\\\\"):
        raise HTTPException(400, "Invalid image name")
    root = exp_dir.resolve()
    image_path = (root / name).resolve()
    if not image_path.is_relative_to(root):
        raise HTTPException(400, "Invalid image name")
    if not image_path.exists():
        image_path = (root / "weights" / name).resolve()
        if not image_path.is_relative_to(root):
            raise HTTPException(400, "Invalid image name")
    if not image_path.exists():
        raise HTTPException(404, f"Image '{name}' not found")

    return FileResponse(path=str(image_path), media_type="image/png")


# ── Projects list ──


@router.get("/training/projects")
def list_training_projects():
    """List available training projects."""
    results = []
    projects_dir = R.PROJECTS
    if projects_dir.exists():
        for proj_dir in sorted(projects_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            config_path = proj_dir / "project.yaml"
            if config_path.exists():
                import yaml
                try:
                    with open(config_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    results.append({
                        "name": proj_dir.name,
                        "display_name": data.get("name", proj_dir.name),
                        "task": data.get("task", "detection"),
                        "model": data.get("backend", {}).get("model", "yolo26n"),
                        "backend": data.get("backend", {}).get("name", "ultralytics"),
                    })
                except Exception:
                    pass
    return results


def _broadcast_log(experiment_id: str, message: str):
    """Send log to all connected WebSocket clients.

    Called from background training threads, which do NOT have a running
    asyncio event loop. We rely on a module-level reference to the main loop
    (captured at startup) and schedule the send via run_coroutine_threadsafe.
    """
    _broadcast(experiment_id, {"type": "log", "message": message, "experiment_id": experiment_id})


def _broadcast_metric(experiment_id: str, metric_data: dict):
    """P2: 推送结构化 per-epoch 指标到所有 WS 客户端。

    前端可据此实时绘制 loss/mAP 曲线，无需轮询 results 端点。
    消息格式：
        {"type": "metric", "experiment_id": "...", "epoch": 5,
         "metrics": {"train/box_loss": 0.42, "metrics/mAP50(B)": 0.85, ...}}
    """
    _broadcast(experiment_id, {
        "type": "metric",
        "experiment_id": experiment_id,
        "epoch": metric_data.get("epoch", 0),
        "metrics": metric_data,
    })


def _broadcast(experiment_id: str, payload: dict):
    """底层广播：向指定实验的所有 WS 客户端发送 JSON 消息。"""
    clients = _ws_clients.get(experiment_id, [])
    loop = _main_loop
    if loop is None:
        # Fallback: try get_event_loop for older Python versions where it
        # still returns the main loop without raising. Best-effort only.
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            return
    for ws in clients[:]:
        try:
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send_json(payload), loop)
            else:
                # No running loop anywhere — drop the message
                pass
        except Exception:
            pass
