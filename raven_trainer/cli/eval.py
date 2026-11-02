"""raven-train eval — Evaluate a trained model on a validation/test dataset.

Usage:
    # Evaluate the best model from a project
    raven-train eval projects/my_project/project.yaml

    # Evaluate a specific model weights file
    raven-train eval projects/my_project/project.yaml --weights best.pt

    # Evaluate with specific dataset config
    raven-train eval project.yaml --data dataset.yaml

Metrics are saved to the experiment record and printed to console.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml as _yaml

from ..adapters import load_adapter
from ..config import R
from ..data_asset.job import JobManager, JobStatus, JobType
from ._models import latest_model_path


@click.command(name="eval")
@click.argument("project_manifest", type=click.Path(exists=True, dir_okay=False))
@click.option("--weights", "-w", default=None, help="Specific weights file (.pt)")
@click.option("--data", "-d", default=None, help="Dataset config YAML (default: from project)")
@click.option("--split", default="val", help="Dataset split to evaluate (val/test)")
@click.option("--device", default="cuda:0", help="Evaluation device")
@click.option("--conf", type=float, default=None, help="Confidence threshold")
@click.option("--imgsz", type=int, default=None, help="Image size")
@click.option("--batch", type=int, default=None, help="Batch size")
def cmd_eval(
    project_manifest: str,
    weights: str | None,
    data: str | None,
    split: str,
    device: str,
    conf: float | None,
    imgsz: int | None,
    batch: int | None,
):
    """Evaluate a trained model on a dataset.

    PROJECT_MANIFEST is a project.yaml from 'raven-train new'.
    """
    project_path = Path(project_manifest).resolve()
    jm = JobManager()
    job = jm.create(
        job_type=JobType.EVAL,
        config={"project_manifest": project_manifest, "weights": weights, "data": data},
        label=f"Evaluate {project_manifest}",
    )
    jm.update_status(job.job_id, JobStatus.RUNNING)
    jm.append_log(job.job_id, f"Starting evaluation: {project_manifest}")
    with open(project_path, encoding="utf-8") as f:
        manifest = _yaml.safe_load(f)

    backend_name = manifest.get("backend", {}).get("name", "ultralytics")
    model_name = manifest.get("backend", {}).get("model", "yolo26n")
    project_name = manifest.get("name", project_path.parent.name)

    # Determine weights path — 兼容 .pt / .pth (P0-3)
    weights_path: Optional[Path]
    if weights:
        weights_path = Path(weights)
    else:
        models_dir = R.models_dir(project_name)
        weights_path = latest_model_path(models_dir)
        if not weights_path:
            click.echo(f"[Eval] No model (.pt/.pth) found in {models_dir}. Train first or specify --weights.")
            jm.update_status(job.job_id, JobStatus.FAILED, error="No model found")
            return

    if not weights_path.exists():
        click.echo(f"[Eval] Weights not found: {weights_path}")
        jm.update_status(job.job_id, JobStatus.FAILED, error="Weights not found")
        return

    # Determine dataset config
    if data:
        data_path = Path(data)
    else:
        data_path = project_path.parent / "dataset.yaml"

    if not data_path.exists():
        click.echo(f"[Eval] Dataset config not found: {data_path}")
        jm.update_status(job.job_id, JobStatus.FAILED, error="Dataset config not found")
        return

    click.echo(f"[Eval] Project:  {project_name}")
    click.echo(f"[Eval] Model:    {model_name}")
    click.echo(f"[Eval] Weights:  {weights_path}")
    click.echo(f"[Eval] Data:     {data_path}")
    click.echo(f"[Eval] Split:    {split}")
    click.echo(f"[Eval] Device:   {device}")

    # Build config
    eval_cfg = {
        "model": model_name,
        "weights": str(weights_path),
    }
    if conf:
        eval_cfg["conf"] = conf
    if imgsz:
        eval_cfg["imgsz"] = imgsz
    if batch:
        eval_cfg["batch"] = batch

    # Load adapter and run evaluation
    adapter_cls = load_adapter(backend_name)
    adapter = adapter_cls(eval_cfg)

    click.echo("[Eval] Running evaluation...")
    try:
        metrics = adapter.evaluate(
            str(data_path),
            weights=str(weights_path),
            split=split,
            device=device,
            conf=conf,
            imgsz=imgsz,
            batch=batch,
            log_func=lambda msg: jm.append_log(job.job_id, msg),
        )
    except Exception as e:
        click.echo(f"[Eval] Evaluation failed: {e}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=str(e))
        return

    if not metrics:
        jm.update_status(job.job_id, JobStatus.FAILED, error="No metrics returned")
        click.echo("[Eval] No metrics returned. The adapter may not support standalone evaluation.")
        click.echo("[Eval] Try running 'raven-train val' through the framework directly.")
        return

    click.echo("\n[Eval] Results:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            click.echo("  {key:30s} {value:.4f}")
        else:
            click.echo("  {key:30s} {value}")

    # Determine eval sequence number up-front so it can be reused for CSV + YAML + report
    exp_dir = R.experiment_dir(project_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(exp_dir.glob("eval_*.yaml"))
    seq = len(existing) + 1

    # Save per-class metrics to CSV if available
    try:
        import csv as _csv
        per_class = {k: v for k, v in metrics.items() if "(B)" in k and k.startswith("metrics/")}
        if per_class:
            csv_path = exp_dir / "eval_{seq:03d}_per_class.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                w = _csv.writer(cf)
                w.writerow(["metric", "value"])
                for k, v in sorted(per_class.items()):
                    w.writerow([k, v])
            click.echo(f"[Eval] Per-class CSV: {csv_path}")
    except Exception:
        pass

    # Save to experiment record
    eval_path = exp_dir / "eval_{seq:03d}.yaml"

    record = {
        "evaluation": "eval_{seq:03d}",
        "created_at": datetime.now().isoformat(),
        "model": model_name,
        "weights": str(weights_path),
        "dataset": str(data_path),
        "split": split,
        "metrics": metrics,
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        _yaml.dump(record, f, default_flow_style=False, sort_keys=False)

    click.echo(f"[Eval] Record saved: {eval_path}")

    jm.save_result(job.job_id, {
        "project": project_name,
        "weights": str(weights_path),
        "dataset": str(data_path),
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    })
    jm.append_log(job.job_id, "Evaluation complete.")
    jm.update_status(job.job_id, JobStatus.COMPLETED)

    # Generate evaluation report
    try:
        from ..report import generate_eval_report
        report_path = exp_dir / "eval_{seq:03d}_report.html"
        generate_eval_report(
            eval_path=eval_path,
            project_name=project_name,
            metrics=metrics,
            model_used=str(weights_path),
            dataset_used=str(data_path),
            output_path=report_path,
        )
        click.echo(f"[Eval] Report: {report_path}")
    except Exception as e:
        click.echo(f"[Eval] Report generation skipped: {e}")
