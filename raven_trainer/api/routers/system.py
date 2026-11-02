"""System API — 系统状态和配置。"""

from __future__ import annotations

from pathlib import Path

import yaml

from fastapi import APIRouter

from ...config import R

router = APIRouter(tags=["system"])


def _count_images_fallback(asset_dir: Path) -> int:
    """Count images by iterating manifest.jsonl files (fallback when cache absent)."""
    total = 0
    versions_dir = asset_dir / "versions"
    if versions_dir.exists():
        for v_dir in versions_dir.iterdir():
            manifest_path = v_dir / "manifest.jsonl"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    total += sum(1 for line in mf if line.strip())
    return total


@router.get("/system/status")
def system_status():
    """Overall system status overview for dashboard."""
    datasets_dir = R.DATASETS
    dataset_count = 0
    total_images = 0
    total_versions = 0

    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir() and (d / "asset.yaml").exists():
                dataset_count += 1
                versions_dir = d / "versions"
                if versions_dir.exists():
                    total_versions += len([v for v in versions_dir.iterdir() if v.is_dir()])
                # M-21: Read cached total_images from asset.yaml
                # Fallback to line-counting if cache missing (e.g. pre-migration)
                try:
                    with open(d / "asset.yaml", encoding="utf-8") as af:
                        asset_data = yaml.safe_load(af) or {}
                    cached = asset_data.get("total_images")
                    if cached is not None:
                        total_images += cached
                    else:
                        total_images += _count_images_fallback(d)
                except Exception:
                    total_images += _count_images_fallback(d)

    # Count models — model files (.pt/.onnx/.engine), NOT subdirectories
    models_count = 0
    projects_dir = R.PROJECTS
    model_exts = {".pt", ".onnx", ".engine"}
    if projects_dir.exists():
        for p in projects_dir.iterdir():
            models_dir = p / "models"
            if models_dir.exists():
                models_count += len([
                    m for m in models_dir.iterdir()
                    if m.is_file() and m.suffix.lower() in model_exts
                ])

    return {
        "version": "0.5.0",
        "data_root": str(R.DATA.resolve()),
        "datasets": dataset_count,
        "total_images": total_images,
        "total_versions": total_versions,
        "models": models_count,
    }


@router.get("/system/activities")
def recent_activities(limit: int = 10):
    """Recent activities across the system."""
    activities = []

    # Check jobs
    jobs_dir = R.DATA / "jobs"
    if jobs_dir.exists():
        import json
        for f in sorted(jobs_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                activities.append({
                    "type": "job",
                    "action": f"{data.get('type', 'job')}: {data.get('job_id', '')}",
                    "status": data.get("status", ""),
                    "time": data.get("created_at", ""),
                })
            except Exception:
                pass

    # Check ingestion runs
    runs_dir = R.INGESTION_RUNS
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:limit]:
            report_json = run_dir / "report.json"
            if report_json.exists():
                import json
                try:
                    with open(report_json, encoding="utf-8") as fp:
                        data = json.load(fp)
                    activities.append({
                        "type": "ingest",
                        "action": f"Ingestion: {data.get('kept_frames', 0)} frames",
                        "status": "completed",
                        "time": data.get("created_at", ""),
                    })
                except Exception:
                    pass

    # Deduplicate by time and sort
    activities.sort(key=lambda x: x.get("time", ""), reverse=True)
    return activities[:limit]
