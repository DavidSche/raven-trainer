"""raven-train ingest — Data Ingestion Pipeline commands."""

from __future__ import annotations

from pathlib import Path

import click

from ..config import R
from ..ingestion.profile import IngestionProfile, IngestionProfileManager, SourceRef
from ..data_asset.job import JobManager, JobStatus, JobType
import time as _t


@click.group(name="ingest")
def cmd_ingest():
    """Data ingestion pipeline (import, sample, clean, dedup, scene, report)."""


@cmd_ingest.command(name="scan")
@click.argument("raw_root", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output metadata index path")
def ingest_scan(raw_root: str, output: str | None):
    """Scan raw/ directory and build metadata index."""
    from ..ingestion.importers import RawImporter

    importer = RawImporter(raw_root)
    sources = importer.scan()
    click.echo(f"[Ingestion] Scanned {len(sources)} sources:")
    for s in sources:
        meta = s.metadata
        if meta:
            click.echo(f"  {meta.source_name:20s}  {meta.source_type:10s}  {meta.resolution or '?':12s}  {meta.total_frames or '?':>8} frames")

    import json
    out_path = Path(output) if output else R.CACHE / "raw_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.metadata.to_dict() for s in sources if s.metadata]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    click.echo(f"[Ingestion] Index saved: {out_path}")


@cmd_ingest.command(name="run")
@click.argument("raw_root", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory (default: <DATA>/ingested)")
@click.option("--pipeline", "-p", default="motion,scene_change,burst,balance",
              help="Sampling pipeline stages (comma-separated)")
@click.option("--interval", type=int, default=30, help="Frame interval")
@click.option("--motion-threshold", type=float, default=0.5, help="Motion area %% threshold (0-100)")
@click.option("--scene-method", default="histogram", type=click.Choice(["histogram", "mse"]),
              help="Scene change detection method")
@click.option("--scene-threshold", type=float, default=0.3, help="Scene change sensitivity")
@click.option("--burst-before", type=int, default=5, help="Frames before event")
@click.option("--burst-after", type=int, default=10, help="Frames after event")
@click.option("--sample-count", type=int, default=500, help="Number of sample positions")
@click.option("--max-frames", type=int, default=5000, help="Max frames per camera")
@click.option("--blur-threshold", type=float, default=100.0, help="Laplacian blur threshold")
@click.option("--dedup-method", default="dhash", type=click.Choice(["dhash", "phash", "ssim"]),
              help="Dedup method")
@click.option("--dedup-distance", type=int, default=5, help="Max hash distance")
@click.option("--scene-analysis", is_flag=True, default=False, help="Enable scene clustering")
@click.option("--no-report", is_flag=True, default=False, help="Skip HTML report")
@click.option("--person-detection", is_flag=True, default=False,
              help="Use YOLO person detection (requires raven-ultralytics)")
@click.option("--decode-mode", default="interval",
              type=click.Choice(["interval", "fps", "keyframe"]),
              help="Frame decode strategy: interval / fps / keyframe")
@click.option("--every-n-frames", type=int, default=30, help="Interval mode: extract every N frames")
@click.option("--target-fps", type=float, default=0.0, help="FPS mode: target frames per second")
@click.option("--jpeg-quality", type=int, default=80, help="JPEG output quality (1-100)")
def ingest_run(
    raw_root: str, output: str | None,
    pipeline: str, interval: int,
    motion_threshold: float,
    scene_method: str, scene_threshold: float,
    burst_before: int, burst_after: int,
    sample_count: int, max_frames: int,
    blur_threshold: float,
    dedup_method: str, dedup_distance: int,
    scene_analysis: bool, no_report: bool,
    person_detection: bool,
    decode_mode: str, every_n_frames: int, target_fps: float,
    jpeg_quality: int,
):
    """Run full ingestion pipeline: scan -> sample -> quality -> dedup -> report -> register."""
    from ..ingestion.core import QualityFilterConfig, DedupConfig
    from ..ingestion.pipeline import IngestionPipeline
    from ..ingestion.sampling import build_pipeline
    from ..ingestion.sampling.core import DecodeConfig

    jm = JobManager()
    job = jm.create(
        job_type=JobType.INGEST,
        config={
            "raw_root": raw_root,
            "pipeline": pipeline,
            "dedup_method": dedup_method,
        },
        label=f"Ingest raw data from {raw_root}",
    )

    try:
        jm.update_status(job.job_id, JobStatus.RUNNING)
        jm.append_log(job.job_id, f"Starting ingestion from {raw_root}")

        stages = [s.strip() for s in pipeline.split(",")]
        if person_detection and "object" not in stages:
            stages.insert(0, "object")

        if output:
            out_path = Path(output)
        else:
            out_path = R.ingestion_run_dir(f"ingest_{_t.strftime('%Y%m%d_%H%M%S')}")

        sampler_config = {
            "interval": {"interval": interval},
            "motion": {"motion_area_threshold": motion_threshold},
            "scene_change": {"method": scene_method, "threshold": scene_threshold},
            "burst": {"before": burst_before, "after": burst_after},
            "balance": {"max_per_camera": max_frames},
        }

        quality = QualityFilterConfig(blur_laplacian_threshold=blur_threshold)
        dedup = DedupConfig(method=dedup_method, hash_distance=dedup_distance)
        decode_config = DecodeConfig(
            mode=decode_mode,
            every_n_frames=every_n_frames,
            target_fps=target_fps,
            jpeg_quality=jpeg_quality,
        )

        R.ensure_dirs()

        pipeline_obj = IngestionPipeline(
            raw_root=raw_root,
            sampling_pipeline=stages,
            sampler_config=sampler_config,
            sample_count=sample_count,
            quality=quality,
            dedup=dedup,
            decode_config=decode_config,
        )

        report = pipeline_obj.run(
            output_dir=out_path,
            skip_scene=not scene_analysis,
            generate_report=not no_report,
        )

        result_dict = {
            "total_after_dedup": report.total_after_dedup,
            "total_sampled": report.total_sampled,
            "total_sources": report.total_sources,
            "quality_removed_blurry": report.quality_removed_blurry,
            "quality_removed_black": report.quality_removed_black,
            "quality_removed_overexposed": report.quality_removed_overexposed,
            "quality_removed_small": report.quality_removed_small,
            "dedup_removed": report.dedup_removed,
            "output_dir": str(out_path),
        }
        jm.save_result(job.job_id, result_dict)
        jm.append_log(job.job_id, f"Done. {report.total_after_dedup} frames kept.")
        jm.update_status(job.job_id, JobStatus.COMPLETED)

        click.echo(f"\n[Ingestion] Done. {report.total_after_dedup} frames kept.")
        click.echo(f"[Ingestion] Output: {out_path}")
        click.echo(f"[Ingestion] Job: {job.job_id}")
        click.echo(f"[Ingestion] Use 'raven-train dataset list' to see registered assets.")

    except Exception as e:
        jm.append_log(job.job_id, f"Ingestion FAILED: {e}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=str(e))
        click.echo(f"\n[Ingestion] FAILED: {e}")
        raise


@cmd_ingest.command(name="report")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output HTML path")
def ingest_report(input_dir: str, output: str | None):
    """Generate HTML quality report from an existing ingestion output."""
    from ..ingestion.report import ReportGenerator

    input_path = Path(input_dir)
    output_path = Path(output) if output else input_path / "ingestion_report.html"
    click.echo(f"[Ingestion] Generating report for: {input_path}")
    click.echo(f"[Ingestion] Report saved: {output_path}")



# ── Ingestion Profile commands ──

@cmd_ingest.group(name="profile")
def cmd_profile():
    """管理 Ingestion Profile（数据摄入配置，可保存、可复现、可管理）。"""


@cmd_profile.command(name="list")
def profile_list():
    """列出所有 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    profiles = mgr.list_profiles()
    if not profiles:
        click.echo("[Ingestion Profile] No profiles found.")
        return
    click.echo(f"{'NAME':25s} {'SOURCES':10s} {'PIPELINE':35s} {'TARGET':20s}")
    click.echo("-" * 90)
    for name in profiles:
        p = mgr.get_profile(name)
        if p:
            src_count = str(len(p.sources))
            pipe = ",".join(p.pipeline)
            target = p.target_dataset or "-"
            click.echo(f"{name:25s} {src_count:10s} {pipe:35s} {target:20s}")


@cmd_profile.command(name="show")
@click.argument("name")
def profile_show(name: str):
    """查看 Ingestion Profile 内容。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        click.echo(f"[Ingestion Profile] '{name}' not found.")
        return
    import yaml as _yaml
    click.echo(_yaml.dump(p.to_dict(), default_flow_style=False, sort_keys=False))


@cmd_profile.command(name="create")
@click.argument("name")
@click.option("--source", "-s", multiple=True, help="Raw source: name:path")
@click.option("--pipeline", "-p", default="motion,scene_change,burst,balance", help="Pipeline stages")
@click.option("--sample-count", type=int, default=500, help="Sample count")
@click.option("--target-dataset", help="Target asset dataset name")
@click.option("--description", "-d", help="Description")
@click.option("--blur-threshold", type=float, default=100.0, help="Laplacian blur threshold")
@click.option("--dedup-method", default="dhash", help="Dedup method")
@click.option("--dedup-distance", type=int, default=5, help="Hash distance")
@click.option("--decode-mode", default="interval",
              type=click.Choice(["interval", "fps", "keyframe"]),
              help="Frame decode strategy")
@click.option("--every-n-frames", type=int, default=30, help="Interval mode: extract every N frames")
@click.option("--target-fps", type=float, default=0.0, help="FPS mode: target frames per second")
@click.option("--jpeg-quality", type=int, default=80, help="JPEG output quality (1-100)")
def profile_create(name, source, pipeline, sample_count, target_dataset, description,
                   blur_threshold, dedup_method, dedup_distance,
                   decode_mode, every_n_frames, target_fps, jpeg_quality):
    """创建 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    existing = mgr.get_profile(name)
    if existing:
        click.echo(f"[Ingestion Profile] '{name}' already exists.")
        return
    sources = []
    for s in source:
        parts = s.split(":", 1)
        src_name = parts[0]
        src_path = parts[1] if len(parts) > 1 else "videos/" + parts[0]
        source_type = "image_dir" if any(src_path.startswith(x) for x in ["images/"]) else "video"
        sources.append(SourceRef(name=src_name, path=src_path, source_type=source_type))
    if not sources:
        click.echo("[Ingestion Profile] At least one --source required (format: name:path)")
        return
    profile = IngestionProfile(
        name=name,
        description=description or "Auto-created profile",
        sources=sources,
        pipeline=[s.strip() for s in pipeline.split(",")],
        sample_count=sample_count,
        target_dataset=target_dataset or name,
        quality={"blur_laplacian_threshold": blur_threshold},
        dedup={"method": dedup_method, "hash_distance": dedup_distance},
        decode={"mode": decode_mode, "every_n_frames": every_n_frames,
               "target_fps": target_fps, "jpeg_quality": jpeg_quality},
    )
    path = mgr.save_profile(profile)
    click.echo(f"[Ingestion Profile] Created: {path}")
    click.echo(f"  Sources: {len(profile.sources)}")
    click.echo(f"  Pipeline: {profile.pipeline}")
    click.echo(f"  Target: {profile.target_dataset}")
    click.echo(f"  Use 'raven-train ingest profile run {name}' to execute")


@cmd_profile.command(name="run")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--no-register", is_flag=True, help="Skip auto-register")
def profile_run(name, output, no_register):
    """执行 Ingestion Profile（Job 跟踪）。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        click.echo(f"[Ingestion Profile] '{name}' not found.")
        return

    jm = JobManager()
    job = jm.create(
        job_type=JobType.INGEST,
        config={
            "profile_name": name,
            "dataset_name": p.target_dataset or name,
        },
        label=f"Ingest profile {name}",
    )

    try:
        jm.update_status(job.job_id, JobStatus.RUNNING)
        jm.append_log(job.job_id, f"Starting ingestion profile: {name}")

        from pathlib import Path
        out_path = Path(output) if output else None
        if no_register:
            p.auto_register = False

        click.echo(f"[Ingestion Profile] Running '{name}'...")
        click.echo(f"  Sources: {len(p.sources)}")
        click.echo(f"  Pipeline: {p.pipeline}")

        report = mgr.run_profile(name, output_dir=out_path)

        result_dict = {
            "total_after_dedup": report.total_after_dedup,
            "total_sampled": report.total_sampled,
            "total_sources": report.total_sources,
            "kept_frames": report.total_after_dedup,
            "quality_removed_blurry": report.quality_removed_blurry,
            "quality_removed_black": report.quality_removed_black,
            "quality_removed_overexposed": report.quality_removed_overexposed,
            "quality_removed_small": report.quality_removed_small,
            "dedup_removed": report.dedup_removed,
            "source_distribution": report.source_distribution,
        }
        jm.save_result(job.job_id, result_dict)
        jm.append_log(job.job_id, f"Done. {report.total_after_dedup} frames kept.")
        jm.update_status(job.job_id, JobStatus.COMPLETED)

        click.echo(f"\n[Ingestion Profile] Done.")
        click.echo(f"  Kept: {report.total_after_dedup} frames")
        click.echo(f"  Job: {job.job_id}")

    except Exception as e:
        jm.append_log(job.job_id, f"Ingestion profile run FAILED: {e}")
        jm.update_status(job.job_id, JobStatus.FAILED, error=str(e))
        click.echo(f"\n[Ingestion Profile] FAILED: {e}")
        raise


@cmd_profile.command(name="delete")
@click.argument("name")
def profile_delete(name):
    """删除 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        click.echo(f"[Ingestion Profile] '{name}' not found.")
        return
    mgr.delete_profile(name)
    click.echo(f"[Ingestion Profile] Deleted: {name}")
