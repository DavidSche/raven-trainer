"""Ingestion API — 数据接入管理。"""

from __future__ import annotations

import json
import threading
import contextlib

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..schemas import IngestionRunSchema, StorageSourceSchema
from ...config import R
from ...ingestion.profile import IngestionProfileManager, IngestionProfile, SourceRef
from ...data_asset.job import JobManager, JobStatus, JobType

router = APIRouter(tags=["ingestion"])

# ── Default profile seeding ──

DEFAULT_PROFILES = [
    {
        "name": "default_video",
        "description": "Default video ingestion: motion detection + scene change + burst capture + quality balance",
        "pipeline": ["motion", "scene_change", "burst", "balance"],
        "sample_count": 500,
        "target_dataset": "",  # user sets this when cloning
        "decode": {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80},
        "quality": {"blur_laplacian_threshold": 100.0, "black_mean_threshold": 5.0,
                     "overexposed_mean_threshold": 250.0, "overexposed_pixel_ratio": 0.5,
                     "min_width": 320, "min_height": 240},
        "dedup": {"method": "dhash", "hash_size": 8, "hash_distance": 5, "ssim_threshold": 0.98},
    },
    {
        "name": "default_image",
        "description": "Default image ingestion: interval sampling + quality filter + deduplication",
        "pipeline": ["balance"],
        "sample_count": 200,
        "target_dataset": "",
        "decode": {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80},
        "quality": {"blur_laplacian_threshold": 200.0, "black_mean_threshold": 3.0,
                     "overexposed_mean_threshold": 245.0, "overexposed_pixel_ratio": 0.3,
                     "min_width": 128, "min_height": 128},
        "dedup": {"method": "phash", "hash_size": 8, "hash_distance": 5, "ssim_threshold": 0.95},
    },
]

# M-41: Track whether the default-profile seeding has already run in this
# process so we don't repeat the (idempotent but I/O-bearing) work on every
# request that triggers the lazy seed.
_default_profiles_seeded = False


def _seed_default_profiles():
    """Create default profiles if they don't exist yet.

    M-41: Previously this was called at module import time, which meant the
    mere act of importing ``api.routers.ingestion`` had side effects
    (writing files to disk, printing to stdout). That breaks tooling that
    imports the module for inspection (linters, type checkers, doc
    generators, tests that don't want disk writes) and couples import to
    a writable data directory. The call is now lazy: it runs the first
    time ``list_ingestion_profiles`` is invoked, which is the actual
    user-facing entry point that depends on the defaults existing.
    """
    global _default_profiles_seeded
    if _default_profiles_seeded:
        return
    _default_profiles_seeded = True
    mgr = IngestionProfileManager()
    existing = set(mgr.list_profiles())
    for cfg in DEFAULT_PROFILES:
        if cfg["name"] in existing:
            continue
        profile = IngestionProfile(
            name=cfg["name"],
            description=cfg["description"],
            pipeline=cfg["pipeline"],
            sample_count=cfg["sample_count"],
            target_dataset=cfg["target_dataset"],
            quality=cfg["quality"],
            dedup=cfg["dedup"],
            decode=cfg.get("decode", {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80}),
        )
        mgr.save_profile(profile)
        print(f"  [Seed] Created default profile: {cfg['name']}")


@router.get("/storage/sources")
def list_storage_sources():
    """List storage sources (videos/images from storage/)."""
    sources = []
    storage_videos = R.STORAGE_VIDEOS
    storage_images = R.STORAGE_IMAGES
    if storage_videos.exists():
        for f in sorted(storage_videos.rglob("*.mp4"))[:100]:
            sources.append(StorageSourceSchema(
                name=f.stem,
                type="video",
                path=str(f.relative_to(R.STORAGE).as_posix()),
            ))
    if storage_images.exists():
        for d in sorted(storage_images.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*")))
                sources.append(StorageSourceSchema(
                    name=d.name,
                    type="directory",
                    path=str(d.relative_to(R.STORAGE).as_posix()),
                    frames=count,
                ))
    return sources


@router.get("/ingestion/runs")
def list_ingestion_runs(limit: int = Query(50)):
    """List all ingestion run records."""
    runs_dir = R.INGESTION_RUNS
    results = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:limit]:
            run_id = run_dir.name
            report_json = run_dir / "report.json"
            report = {}
            if report_json.exists():
                with open(report_json, encoding="utf-8") as f:
                    report = json.load(f)
            results.append(IngestionRunSchema(
                run_id=run_id,
                status=report.get("status", "completed"),
                source_count=report.get("source_count", 0),
                total_frames=report.get("total_frames", 0),
                kept_frames=report.get("kept_frames", 0),
                created_at=report.get("created_at", ""),
                pipeline=report.get("pipeline", ""),
                report_path=str(report_json.relative_to(R.DATA.parent)) if report_json.exists() else "",
            ))
    return results


@router.get("/ingestion/runs/{run_id}/report")
def get_ingestion_report(run_id: str):
    run_dir = R.ingestion_run_dir(run_id)
    report_html = run_dir / "report.html"
    if report_html.exists():
        return {"html": report_html.read_text(encoding="utf-8")}
    report_json = run_dir / "report.json"
    if report_json.exists():
        return {"json": json.loads(report_json.read_text(encoding="utf-8"))}
    raise HTTPException(404, f"Report for run '{run_id}' not found")

# ── Ingestion Profile API ──


class CreateProfileRequest(BaseModel):
    name: str
    description: str = ""
    sources: list = []
    pipeline: list = ["motion", "scene_change", "burst", "balance"]
    sample_count: int = 500
    target_dataset: str = ""
    blur_threshold: float = 100.0
    dedup_method: str = "dhash"
    dedup_distance: int = 5
    decode: dict = {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80}


@router.get("/ingestion/profiles")
def list_ingestion_profiles():
    """列出所有 Ingestion Profile，包含 last_run 信息。"""
    # M-41: Seed defaults lazily on the first user-facing list call rather
    # than at module import, so importing this router has no side effects.
    _seed_default_profiles()
    mgr = IngestionProfileManager()
    names = mgr.list_profiles()
    jm = JobManager()
    # Build a map: profile_name → latest job info
    profile_jobs: dict[str, dict] = {}
    for job in jm.list_jobs(limit=200):
        if job.type != JobType.INGEST:
            continue
        pn = job.config.get("profile_name", "")
        if not pn:
            continue
        # Keep the most recent job per profile
        existing = profile_jobs.get(pn)
        if existing is None or job.created_at > existing["created_at"]:
            profile_jobs[pn] = {
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at,
            }
    profiles = []
    for name in names:
        p = mgr.get_profile(name)
        if p:
            d = p.to_dict()
            profiles.append({
                "name": d["name"],
                "description": d["description"],
                "source_count": len(d["sources"]),
                "pipeline": d["pipeline"],
                "target_dataset": d["target_dataset"],
                "created_at": d["created_at"],
                "sources": d["sources"],
                "last_run": profile_jobs.get(name),
            })
    return {"profiles": profiles}


@router.get("/ingestion/profiles/{name}")
def get_ingestion_profile(name: str):
    """获取 Ingestion Profile 详情。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        raise HTTPException(404, f"IngestionProfile '{name}' not found")
    return p.to_dict()


@router.post("/ingestion/profiles")
def create_ingestion_profile(req: CreateProfileRequest):
    """创建 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    existing = mgr.get_profile(req.name)
    if existing:
        raise HTTPException(409, f"IngestionProfile '{req.name}' already exists")

    sources = []
    for s in req.sources:
        if isinstance(s, dict):
            sources.append(SourceRef(
                name=s.get("name", ""),
                path=s.get("path", ""),
                source_type=s.get("source_type", "video"),
            ))
        elif isinstance(s, str):
            sources.append(SourceRef(name=s, path="videos/" + s))

    profile = IngestionProfile(
        name=req.name,
        description=req.description,
        sources=sources,
        pipeline=req.pipeline,
        sample_count=req.sample_count,
        target_dataset=req.target_dataset or req.name,
        quality={"blur_laplacian_threshold": req.blur_threshold},
        dedup={"method": req.dedup_method, "hash_distance": req.dedup_distance},
        decode=req.decode,
    )
    path = mgr.save_profile(profile)
    return {"status": "created", "path": str(path), "profile": profile.to_dict()}


def _run_profile_thread(job_id: str, profile_name: str):
    jm = JobManager()
    mgr = IngestionProfileManager()

    try:
        jm.update_status(job_id, JobStatus.RUNNING)
        jm.append_log(job_id, f"Starting Ingestion Profile execution: {profile_name}")

        profile = mgr.get_profile(profile_name)
        if not profile:
            raise ValueError(f"IngestionProfile '{profile_name}' not found")

        from ...config import R as _R
        run_dir = _R.ingestion_run_dir(f"profile_{profile_name}")

        # Redirect standard stdout/stderr streams to job log file
        job = jm.get(job_id)
        if job and job.log_path:
            with open(job.log_path, "a", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    report = mgr.run_profile(profile_name, output_dir=run_dir)
        else:
            report = mgr.run_profile(profile_name, output_dir=run_dir)

        result_dict = {
            "status": "completed",
            "total_sources": report.total_sources,
            "total_sampled": report.total_sampled,
            "total_kept_after_quality": report.total_kept_after_quality,
            "total_after_dedup": report.total_after_dedup,
            "kept_frames": report.total_after_dedup,
            "quality_removed_blurry": report.quality_removed_blurry,
            "quality_removed_black": report.quality_removed_black,
            "quality_removed_overexposed": report.quality_removed_overexposed,
            "quality_removed_small": report.quality_removed_small,
            "dedup_removed": report.dedup_removed,
            "source_distribution": report.source_distribution,
        }
        jm.save_result(job_id, result_dict)
        jm.append_log(job_id, f"Ingestion completed. Kept {report.total_after_dedup} frames.")
        jm.update_status(job_id, JobStatus.COMPLETED)

    except Exception as e:
        error_msg = f"Run failed: {e}\n"
        import traceback
        error_msg += traceback.format_exc()
        jm.append_log(job_id, error_msg)
        jm.update_status(job_id, JobStatus.FAILED, error=str(e))


@router.post("/ingestion/profiles/{name}/run")
def run_ingestion_profile(name: str):
    """执行 Ingestion Profile（异步 Job 模式）。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        raise HTTPException(404, f"IngestionProfile '{name}' not found")

    jm = JobManager()
    target_ds = p.target_dataset or name
    job = jm.create(
        job_type=JobType.INGEST,
        config={
            "profile_name": name,
            "dataset_name": target_ds,
        },
        label=f"Ingest raw data via profile {name} -> {target_ds}",
    )

    thread = threading.Thread(
        target=_run_profile_thread,
        args=(job.job_id, name),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "job_id": job.job_id,
        "message": f"Ingestion job {job.job_id} started. Track logs on jobs page.",
    }


@router.delete("/ingestion/profiles/{name}")
def delete_ingestion_profile(name: str):
    """删除 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        raise HTTPException(404, f"IngestionProfile '{name}' not found")
    mgr.delete_profile(name)
    return {"status": "deleted", "name": name}


@router.put("/ingestion/profiles/{name}")
def update_ingestion_profile(name: str, req: CreateProfileRequest):
    """更新 Ingestion Profile。"""
    mgr = IngestionProfileManager()
    existing = mgr.get_profile(name)
    if existing is None:
        raise HTTPException(404, f"IngestionProfile '{name}' not found")

    sources = []
    for s in req.sources:
        if isinstance(s, dict):
            sources.append(SourceRef(
                name=s.get("name", ""),
                path=s.get("path", ""),
                source_type=s.get("source_type", "video"),
            ))
        elif isinstance(s, str):
            sources.append(SourceRef(name=s, path="videos/" + s))

    profile = IngestionProfile(
        name=name,
        description=req.description,
        sources=sources,
        pipeline=req.pipeline,
        sample_count=req.sample_count,
        target_dataset=req.target_dataset or name,
        quality={**existing.quality, "blur_laplacian_threshold": req.blur_threshold},
        dedup={**existing.dedup, "method": req.dedup_method, "hash_distance": req.dedup_distance},
        created_at=existing.created_at,
        decode=req.decode,
    )
    path = mgr.save_profile(profile)
    return {"status": "updated", "path": str(path), "profile": profile.to_dict()}


@router.post("/ingestion/profiles/{name}/clone")
def clone_ingestion_profile(name: str, req: CreateProfileRequest):
    """克隆 Ingestion Profile 为新名称（Save As）。"""
    if not req.name or req.name == name:
        raise HTTPException(400, "New profile name must be different from source")

    mgr = IngestionProfileManager()
    existing = mgr.get_profile(req.name)
    if existing:
        raise HTTPException(409, f"IngestionProfile '{req.name}' already exists")

    source = mgr.get_profile(name)
    if source is None:
        raise HTTPException(404, f"IngestionProfile '{name}' not found")

    # Deep-copy source with new name
    d = source.to_dict()
    sources = []
    for s in req.sources if req.sources else d.get("sources", []):
        if isinstance(s, dict):
            sources.append(SourceRef(
                name=s.get("name", ""),
                path=s.get("path", ""),
                source_type=s.get("source_type", "video"),
            ))
        elif isinstance(s, str):
            sources.append(SourceRef(name=s, path="videos/" + s))

    profile = IngestionProfile(
        name=req.name,
        description=req.description or d.get("description", ""),
        sources=sources,
        pipeline=req.pipeline if req.pipeline != ["motion", "scene_change", "burst", "balance"] else d.get("pipeline", req.pipeline),
        sample_count=req.sample_count if req.sample_count != 500 else d.get("sample_count", req.sample_count),
        target_dataset=req.target_dataset or d.get("target_dataset", "") or req.name,
        decode=req.decode or d.get("decode", {"mode": "interval", "every_n_frames": 30, "target_fps": 0.0, "jpeg_quality": 80}),
    )
    path = mgr.save_profile(profile)
    return {"status": "cloned", "path": str(path), "profile": profile.to_dict()}
