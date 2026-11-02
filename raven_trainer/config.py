"""Path constants for Raven Trainer (v2.2).

三层级:
  Layer 1 - Raw Data:            storage/           ← 原始数据（Immutable）
  Layer 2 - Asset Datasets:     datasets/           ← 数据资产（版本化、带标注）
  Layer 3 - Training Workspace: workspace/          ← 训练工作区（Recipe 的物化产物）

Recipe（构建配方）独立于三者之外:
  recipes/     ← 构建规则（Asset Dataset → Workspace 的桥梁）

Override RAVEN_DATA env to change data root.
"""

from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    env = os.environ.get("RAVEN_DATA")
    if env:
        return Path(env)
    cwd_data = Path("data")
    if cwd_data.exists():
        return cwd_data
    return _PROJECT_ROOT / "data"


class R:
    """Path constants."""

    # ── Root ──
    DATA: Path = _data_root()
    PROJECT_ROOT: Path = _PROJECT_ROOT

    # ── Layer 1: Raw Data (Immutable) ──
    STORAGE: Path = DATA / "storage"
    STORAGE_IMAGES: Path = STORAGE / "images"
    STORAGE_VIDEOS: Path = STORAGE / "videos"

    # Ingestion run records and profiles
    INGESTION: Path = DATA / "ingestion"
    INGESTION_RUNS: Path = INGESTION / "runs"
    INGESTION_PROFILES: Path = INGESTION / "profiles"

    # ── Layer 2: Asset Datasets (versioned, annotated) ──
    DATASETS: Path = DATA / "datasets"

    # ── Recipes (standalone; many-to-many bridge between datasets and workspaces) ──
    RECIPES: Path = DATA / "recipes"

    # ── Layer 3: Training Workspaces (materialized from Recipe) ──
    WORKSPACE: Path = DATA / "workspace"

    # ── Projects (business projects with experiments + models) ──
    PROJECTS: Path = _PROJECT_ROOT / "projects"

        # ── Pipeline Profiles ──
    PIPELINE_PROFILES: Path = DATA / "pipeline" / "profiles"

    # ── Cache ──
    CACHE: Path = DATA / "cache"

    @classmethod
    def ensure_dirs(cls) -> None:
        for attr in ["STORAGE", "STORAGE_IMAGES", "STORAGE_VIDEOS",
                      "INGESTION", "INGESTION_RUNS", "INGESTION_PROFILES",
                      "DATASETS", "RECIPES", "WORKSPACE", "CACHE",
                      "PROJECTS", "PIPELINE_PROFILES"]:
            path = getattr(cls, attr)
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve_storage_path(cls, storage_path: str) -> Path:
        p = Path(storage_path)
        if p.is_absolute():
            if p.exists():
                return p
            raise FileNotFoundError(f"Storage path not found: {p}")
        candidate = cls.PROJECT_ROOT / p
        if candidate.exists():
            return candidate
        if not str(p).startswith("data/"):
            candidate = cls.DATA / p
            if candidate.exists():
                return candidate
        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            return cwd_candidate
        raise FileNotFoundError(
            f"Storage path not found: {storage_path}"
        )

    # ── Recipe helpers (standalone, not tied to any single dataset) ──
    @classmethod
    def recipe_dir(cls, name: str) -> Path:
        return cls.RECIPES / name

    @classmethod
    def recipe_yaml_path(cls, name: str) -> Path:
        return cls.RECIPES / name / "recipe.yaml"

    # ── Domain Asset helpers ──
    @classmethod
    def dataset_dir(cls, name: str) -> Path:
        return cls.DATASETS / name

    @classmethod
    def dataset_version_dir(cls, dataset_name: str, version: str) -> Path:
        return cls.DATASETS / dataset_name / "versions" / version

    @classmethod
    def dataset_manifest_path(cls, dataset_name: str, version: str) -> Path:
        return cls.dataset_version_dir(dataset_name, version) / "manifest.jsonl"

    @classmethod
    def ingestion_run_dir(cls, run_id: str) -> Path:
        return cls.INGESTION_RUNS / run_id

    # ── Training Asset helpers ──
    @classmethod
    def workspace_dir(cls, name: str) -> Path:
        return cls.WORKSPACE / name

    @classmethod
    def project_dir(cls, name: str) -> Path:
        return cls.PROJECTS / name

    @classmethod
    def experiment_dir(cls, project_name: str) -> Path:
        return cls.project_dir(project_name) / "experiments"

    @classmethod
    def models_dir(cls, project_name: str) -> Path:
        return cls.project_dir(project_name) / "models"

