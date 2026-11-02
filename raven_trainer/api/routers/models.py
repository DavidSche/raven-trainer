"""Models API — 模型管理。

Phase 2: 完整的模型详情、元数据、下载功能。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..schemas import ModelSchema
from ...config import R
from ...experiments import scan_all_experiments

router = APIRouter(tags=["models"])


def _scan_models() -> List[Dict[str, Any]]:
    """Scan projects directory for exported models."""
    results = []
    projects_dir = R.PROJECTS
    if not projects_dir.exists():
        return results
    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        models_dir = proj_dir / "models"
        if not models_dir.exists():
            continue
        for model_file in sorted(models_dir.iterdir()):
            if model_file.suffix not in ('.pt', '.onnx', '.engine'):
                continue
            meta_file = model_file.with_suffix('.metadata.yaml')
            meta = {}
            if meta_file.exists():
                import yaml
                with open(meta_file, encoding='utf-8') as f:
                    meta = yaml.safe_load(f) or {}
            results.append({
                "id": f"{proj_dir.name}_{model_file.stem}",
                "project": proj_dir.name,
                "version": meta.get("version", model_file.stem),
                "type": meta.get("task", "detection"),
                "format": "pytorch" if model_file.suffix == '.pt' else model_file.suffix[1:],
                "path": str(model_file),
                "metrics": meta.get("metrics", {}),
                "created_at": meta.get("created_at", ""),
                "experiment_id": meta.get("experiment", ""),
            })
    return results


@router.get("/models")
async def list_models(limit: int = Query(50)):
    models = _scan_models()
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return [ModelSchema(**m) for m in models[:limit]]


@router.get("/models/stats")
async def model_stats():
    """Aggregated model statistics."""
    models = _scan_models()
    by_project: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    total_size = 0
    for m in models:
        p = m["project"]
        by_project[p] = by_project.get(p, 0) + 1
        f = m["format"]
        by_format[f] = by_format.get(f, 0) + 1
        mp = Path(m["path"])
        if mp.exists():
            total_size += mp.stat().st_size
    return {
        "total": len(models),
        "by_project": by_project,
        "by_format": by_format,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    models = _scan_models()
    for m in models:
        if m["id"] == model_id:
            model = ModelSchema(**m)
            exp_info = {}
            if model.experiment_id:
                for e in scan_all_experiments():
                    if e.experiment_id == model.experiment_id:
                        exp_info = {
                            "experiment_id": e.experiment_id,
                            "project": e.project,
                            "status": e.status,
                            "metrics": e.metrics,
                            "created_at": e.created_at,
                        }
                        break
            model_path = Path(model.path)
            file_info = {
                "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
                "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else 0,
                "format": model.format,
            }
            return {
                "model": model.model_dump(),
                "experiment": exp_info,
                "file_info": file_info,
            }
    raise HTTPException(404, f"Model '{model_id}' not found")


@router.get("/models/{model_id}/metadata")
async def get_model_metadata(model_id: str):
    """Get model metadata file content."""
    models = _scan_models()
    for m in models:
        if m["id"] == model_id:
            model_path = Path(m["path"])
            meta_file = model_path.with_suffix('.metadata.yaml')
            if meta_file.exists():
                import yaml
                with open(meta_file, encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            return {}
    raise HTTPException(404, f"Model '{model_id}' not found")


@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    models = _scan_models()
    for m in models:
        if m["id"] == model_id:
            model_path = Path(m["path"])
            if model_path.exists():
                return FileResponse(
                    str(model_path),
                    filename=model_path.name,
                    media_type="application/octet-stream",
                )
            raise HTTPException(404, "Model file not found on disk")
    raise HTTPException(404, f"Model '{model_id}' not found")
