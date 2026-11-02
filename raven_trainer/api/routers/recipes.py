"""Recipes API — 独立 Recipe 管理。

Recipe 是独立存储的构建配方（data/recipes/<name>/recipe.yaml），
不绑定到任何单个资产数据集。支持多对多映射。
"""

from __future__ import annotations


import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from ...data_asset.recipe import CatalogRecipeManager, DatasetRecipe, RecipeSource
from ...config import R
from pydantic import BaseModel

router = APIRouter(tags=["recipes"])

# Safe name pattern: letters, digits, underscore, hyphen, dot (no path separators or ..)
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _validate_safe_name(name: str, label: str = "name") -> None:
    """Reject names that could enable path traversal.

    Allows only alphanumeric, underscore, hyphen, and dot. Rejects empty
    names, names containing path separators, and ``..`` sequences.
    """
    if not name:
        raise HTTPException(400, f"Invalid {label}: must not be empty")
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(400, f"Invalid {label}: only letters, digits, '_', '-', '.' are allowed")
    if ".." in name:
        raise HTTPException(400, f"Invalid {label}: '..' is not allowed")


class CreateRecipeRequest(BaseModel):
    name: str
    description: str = ""
    sources: list[dict] = []


class RecipeResponse(BaseModel):
    name: str
    description: str = ""
    sources: list[dict] = []
    sourceCount: int = 0
    createdAt: str = ""


def _to_response(recipe: DatasetRecipe) -> RecipeResponse:
    return RecipeResponse(
        name=recipe.name,
        description=recipe.description,
        sources=[{"name": s.name, "path": s.path} for s in recipe.sources],
        sourceCount=len(recipe.sources),
        createdAt=recipe.created_at,
    )


@router.get("/recipes")
async def list_recipes():
    """列出所有独立 Recipe。"""
    mgr = CatalogRecipeManager()
    names = mgr.list_recipes()
    recipes = []
    for name in names:
        recipe = mgr.get_recipe(name)
        if recipe:
            recipes.append(_to_response(recipe))
    return {"recipes": recipes}


@router.get("/recipes/{name}")
async def get_recipe(name: str):
    """获取单个 Recipe 详情。"""
    mgr = CatalogRecipeManager()
    recipe = mgr.get_recipe(name)
    if recipe is None:
        raise HTTPException(404, f"Recipe '{name}' not found")
    return _to_response(recipe)


@router.post("/recipes")
async def create_recipe(req: CreateRecipeRequest):
    """创建新 Recipe。"""
    mgr = CatalogRecipeManager()
    existing = mgr.get_recipe(req.name)
    if existing:
        raise HTTPException(409, f"Recipe '{req.name}' already exists")

    sources = []
    for s in req.sources:
        if isinstance(s, dict):
            sources.append(RecipeSource(
                name=s.get("name", ""),
                path=s.get("path"),
                include=s.get("include", []),
                exclude=s.get("exclude", []),
            ))
        elif isinstance(s, str):
            sources.append(RecipeSource(name=s))

    recipe = DatasetRecipe(
        name=req.name,
        description=req.description,
        sources=sources,
        created_at=datetime.now().isoformat(),
    )
    path = mgr.save_recipe(recipe)
    return {"status": "created", "path": str(path), "recipe": _to_response(recipe)}


@router.put("/recipes/{name}")
async def update_recipe(name: str, req: CreateRecipeRequest):
    """更新已有 Recipe。"""
    mgr = CatalogRecipeManager()
    existing = mgr.get_recipe(name)
    if existing is None:
        raise HTTPException(404, f"Recipe '{name}' not found")

    sources = []
    for s in req.sources:
        if isinstance(s, dict):
            sources.append(RecipeSource(
                name=s.get("name", ""),
                path=s.get("path"),
                include=s.get("include", []),
                exclude=s.get("exclude", []),
            ))
        elif isinstance(s, str):
            sources.append(RecipeSource(name=s))

    recipe = DatasetRecipe(
        name=name,
        description=req.description,
        sources=sources,
        created_at=existing.created_at,
    )
    path = mgr.save_recipe(recipe, name)
    return {"status": "updated", "path": str(path), "recipe": _to_response(recipe)}


@router.delete("/recipes/{name}")
async def delete_recipe(name: str):
    """删除 Recipe。"""
    _validate_safe_name(name, "recipe name")
    mgr = CatalogRecipeManager()
    existing = mgr.get_recipe(name)
    if existing is None:
        raise HTTPException(404, f"Recipe '{name}' not found")
    mgr.delete_recipe(name)
    return {"status": "deleted", "name": name}


@router.post("/recipes/{name}/build")
async def build_from_standalone_recipe(name: str, force: bool = False):
    """从独立 Recipe 构建训练工作区（异步 Job 模式）。"""
    from ...data_asset.profile import BuildProfile
    from ...data_asset.job import JobManager, JobStatus, JobType
    import threading, contextlib, traceback as _tb

    mgr = CatalogRecipeManager()
    recipe = mgr.get_recipe(name)
    if recipe is None:
        raise HTTPException(404, f"Recipe '{name}' not found")

    jm = JobManager()
    job = jm.create(
        job_type=JobType.BUILD,
        config={
            "recipe_name": name,
            "force": force,
        },
        label=f"Build workspace from recipe {name}",
    )

    def _build_thread(job_id: str, recipe_name: str):
        jm_local = JobManager()
        try:
            jm_local.update_status(job_id, JobStatus.RUNNING)
            jm_local.append_log(job_id, f"Starting build from recipe: {recipe_name}")

            profile = BuildProfile.detection_default(name=recipe_name)
            from ...data_asset.workspace import WorkspaceBuilder
            builder = WorkspaceBuilder(dataset_name=recipe_name)

            job = jm_local.get(job_id)
            if job and job.log_path:
                with open(job.log_path, "a", encoding="utf-8") as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        result = builder.build_from_catalog(
                            recipe_name=recipe_name,
                            profile=profile,
                            force=force,
                        )
            else:
                result = builder.build_from_catalog(
                    recipe_name=recipe_name,
                    profile=profile,
                    force=force,
                )

            result_dict = {
                "name": result.name,
                "output_dir": str(result.output_dir),
                "dataset_yaml": str(result.dataset_yaml),
                "total_images": result.total_images,
                "train_count": result.train_count,
                "val_count": result.val_count,
                "test_count": result.test_count,
                "class_count": result.class_count,
                "class_names": result.class_names,
                "source_versions": result.source_versions,
                "skipped_blurry": result.skipped_blurry,
                "skipped_empty": result.skipped_empty,
                "skipped_duplicate": result.skipped_duplicate,
                "duration_sec": result.duration_sec,
                "created_at": result.created_at,
            }
            jm_local.save_result(job_id, result_dict)
            jm_local.append_log(
                job_id,
                f"Build complete: {result.train_count} train, {result.val_count} val, "
                f"{result.test_count} test ({result.duration_sec}s)",
            )
            jm_local.update_status(job_id, JobStatus.COMPLETED)

        except Exception as e:
            error_msg = f"Build failed: {e}\n" + _tb.format_exc()
            jm_local.append_log(job_id, error_msg)
            jm_local.update_status(job_id, JobStatus.FAILED, error=str(e))

    thread = threading.Thread(target=_build_thread, args=(job.job_id, name), daemon=True)
    thread.start()

    return {
        "status": "started",
        "job_id": job.job_id,
        "message": f"Build job {job.job_id} started. Track logs on jobs page.",
    }


# ── Workspace Management ──

import json as _json
from pathlib import Path as _Path

@router.get("/workspaces")
async def list_workspaces():
    """列出所有已构建的训练工作区。"""
    from ...config import R as _R
    ws_dir = _R.WORKSPACE
    if not ws_dir.exists():
        return {"workspaces": []}
    workspaces = []
    for d in sorted(ws_dir.iterdir()):
        if d.is_dir():
            ds_yaml = d / "dataset.yaml"
            info = {
                "name": d.name,
                "path": str(d),
                "has_dataset_yaml": ds_yaml.exists(),
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
            }
            if ds_yaml.exists():
                import yaml as _yaml
                try:
                    cfg = _yaml.safe_load(ds_yaml.read_text(encoding="utf-8"))
                    if cfg:
                        info["train_count"] = len(list((d / "train" / "images").glob("*"))) if (d / "train" / "images").exists() else 0
                        info["val_count"] = len(list((d / "val" / "images").glob("*"))) if (d / "val" / "images").exists() else 0
                        if (d / "test" / "images").exists():
                            info["test_count"] = len(list((d / "test" / "images").glob("*")))
                        info["nc"] = cfg.get("nc", 0)
                except Exception:
                    pass
            workspaces.append(info)
    return {"workspaces": workspaces}


@router.delete("/workspaces/{name}")
async def delete_workspace(name: str):
    """删除训练工作区。"""
    _validate_safe_name(name, "workspace name")
    from ...config import R as _R
    import shutil
    ws_root = _R.WORKSPACE.resolve()
    ws_dir = (_R.WORKSPACE / name).resolve()
    # Defense-in-depth: ensure resolved path is still under WORKSPACE
    try:
        ws_dir.relative_to(ws_root)
    except ValueError:
        raise HTTPException(400, "Path traversal detected")
    if not ws_dir.exists():
        raise HTTPException(404, f"Workspace '{name}' not found")
    shutil.rmtree(ws_dir)
    return {"status": "deleted", "name": name}
