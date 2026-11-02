"""raven-train export — Export trained model to ONNX / TensorRT.

Usage:
    # Export from a project
    raven-train export projects/person_intrusion/project.yaml

    # Export with specific format
    raven-train export projects/person_intrusion/project.yaml --format onnx

    # Export with custom output path
    raven-train export projects/person_intrusion/project.yaml --output best.onnx
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import yaml as _yaml

from ..adapters import load_adapter
from ..config import R
from ..data_asset.job import JobManager, JobStatus, JobType
from ._models import latest_model_path, list_model_versions


@click.command(name="export")
@click.argument("project_manifest", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "-f", default="onnx", help="Export format (onnx, tensorrt)")
@click.option("--device", default="cuda:0", help="Export device")
@click.option("--output", "-o", default=None, help="Output path")
@click.option("--imgsz", type=int, default=None, help="Override image size")
@click.option("--half", is_flag=True, help="FP16 quantization")
@click.option("--int8", is_flag=True, help="INT8 quantization (requires dataset.yaml for calibration)")
@click.option("--register", "register_to_platform", is_flag=True,
              help="Register exported ONNX model to raven-go platform")
def cmd_export(
    project_manifest: str,
    format: str,
    device: str,
    output: str | None,
    imgsz: int | None,
    half: bool,
    int8: bool,
    register_to_platform: bool,
):
    """Export trained model from PROJECT_MANIFEST to deployment format."""
    project_path = Path(project_manifest).resolve()
    jm = JobManager()
    job = jm.create(
        job_type=JobType.EXPORT,
        config={"project_manifest": project_manifest, "format": format, "output": output},
        label=f"Export {project_manifest}",
    )
    jm.update_status(job.job_id, JobStatus.RUNNING)
    jm.append_log(job.job_id, f"Starting export: {project_manifest}")

    with open(project_path, encoding="utf-8") as f:
        manifest = _yaml.safe_load(f)

    backend_name = manifest.get("backend", {}).get("name", "ultralytics")
    model_name = manifest.get("backend", {}).get("model", "yolo26n")
    project_name = manifest.get("name", project_path.parent.name)

    models_dir = R.models_dir(project_name)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        # Find latest model version
        existing = list_model_versions(models_dir)
        if existing:
            latest_version = existing[-1]
            onnx_name = latest_version.with_suffix(".onnx").name
        else:
            onnx_name = "v1.0.onnx"
        out_path = models_dir / onnx_name

    # Locate weights file (input to export) — 兼容 .pt / .pth (P0-3)
    weights_path = latest_model_path(models_dir)
    if not weights_path:
        click.echo(f"[Export] No model (.pt/.pth) found in {models_dir}. Train first.")
        jm.update_status(job.job_id, JobStatus.FAILED, error="No model found")
        return

    click.echo(f"[Export] Project:  {project_name}")
    click.echo(f"[Export] Model:    {model_name}")
    click.echo(f"[Export] Weights:  {weights_path}")
    click.echo(f"[Export] Format:   {format}")
    click.echo(f"[Export] Output:   {out_path}")

    adapter_cls = load_adapter(backend_name)
    config = {"model": model_name, "weights": str(weights_path)}
    if imgsz:
        config["imgsz"] = imgsz
    if half:
        config["half"] = half

    adapter = adapter_cls(config)

    # P2-16: metadata 版本号从权重文件名推导（v1.0 -> 1.0），不再硬编码
    weights_version = weights_path.stem.lstrip("v") if weights_path else "1.0"

    # INT8 量化需要校准数据集，从项目目录查找 dataset.yaml
    dataset_yaml_path = project_path.parent / "dataset.yaml"
    dataset_yaml_str = str(dataset_yaml_path) if dataset_yaml_path.exists() else None

    try:
        # adapter.export() returns a Dict[str, Any] with at least "onnx_path"
        # (see UltralyticsAdapter.export in adapters/builtin.py)
        export_result = adapter.export(
            str(out_path),
            fmt=format,
            device=device,
            model_path=str(weights_path),
            # 仅显式指定时传 imgsz；否则交给适配器默认（rfdetr 用模型自身 resolution）
            **({"imgsz": imgsz} if imgsz else {}),
            half=half,
            int8=int8,
            dataset_yaml=dataset_yaml_str,
            metadata={"model_name": model_name, "model_version": weights_version},
            log_func=lambda msg: jm.append_log(job.job_id, msg),
        )
    except Exception as e:
        click.echo(f"[Export] Export failed: {e}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=str(e))
        return

    # Resolve actual onnx path from export result
    if isinstance(export_result, dict):
        onnx_path_str = export_result.get("onnx_path") or export_result.get("path") or ""
    else:
        onnx_path_str = str(export_result)

    if not onnx_path_str or not Path(onnx_path_str).exists():
        click.echo(f"[Export] Export may have failed. Result: {export_result}")
        jm.update_status(job.job_id, JobStatus.FAILED, error="Export produced no file")
        return

    exported_path = Path(onnx_path_str)
    # Copy to the canonical out_path if different
    import shutil
    if exported_path.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(exported_path), str(out_path))
    else:
        out_path = exported_path

    jm.save_result(job.job_id, {
        "project": project_name,
        "output": str(out_path),
        "format": format,
        "verification": export_result.get("verification") if isinstance(export_result, dict) else None,
    })
    jm.append_log(job.job_id, f"Model exported: {out_path}")
    jm.update_status(job.job_id, JobStatus.COMPLETED)
    click.echo(f"[Export] Model exported: {out_path}")

    # Also update or create metadata
    meta_path = out_path.with_suffix(".metadata.yaml")
    meta = {
        "model": model_name,
        "format": format,
        "exported_at": datetime.now().isoformat(),
        "project": project_name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        _yaml.dump(meta, f, default_flow_style=False)
    click.echo(f"[Export] Metadata: {meta_path}")

    # Generate export report
    try:
        from ..report import generate_export_report
        from ..asset_manager import AssetManager as _AM
        report_path = models_dir / f"{out_path.stem}_export_report.html"
        am = _AM()
        sha256 = am.compute_hash(str(out_path))
        generate_export_report(
            project_name=project_name,
            model_name=model_name,
            format_type=format,
            output_path=report_path,
            metadata={"sha256": sha256, "format": format, "imgsz": str(imgsz) if imgsz else "default"},
            out_file=str(out_path),
        )
        click.echo(f"[Export] Report: {report_path}")
    except Exception as e:
        click.echo(f"[Export] Report generation skipped: {e}")

    # P2: 导出完成后注册 ONNX 模型到 raven-go 推理平台
    if register_to_platform and format == "onnx":
        try:
            from ..integration.raven_go import register_model_to_platform
            # 从 project.yaml 提取 task
            task = manifest.get("task", "detection")
            click.echo(f"[Export] Registering model to raven-go platform...")
            reg_result = register_model_to_platform(
                onnx_path=str(out_path),
                model_name=model_name,
                task=task,
                project_name=project_name,
                metrics={},
                quantization={"half": half, "int8": int8},
                log_func=lambda msg: click.echo(msg),
            )
            if reg_result.get("success"):
                click.echo(f"[Export] Model registered to platform: {reg_result.get('model_id', 'unknown')}")
            else:
                click.echo(f"[Export] Platform registration failed: {reg_result.get('error', 'unknown')}")
        except Exception as e:
            click.echo(f"[Export] Platform registration skipped: {e}")
    elif register_to_platform and format != "onnx":
        click.echo(f"[Export] Platform registration skipped: only onnx format is supported (got {format})")
