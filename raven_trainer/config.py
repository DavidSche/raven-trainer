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
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class InvalidNameError(ValueError):
    """Raised when a user-supplied name (dataset/version/project/...) is unsafe.

    Names are used to build filesystem paths all over the codebase, so they
    must never contain path separators, ``..`` segments, drive letters, or
    control characters.  The API layer maps this error to HTTP 400.
    """


# Characters that must never appear in a path component on any platform.
_FORBIDDEN_NAME_CHARS = frozenset('/\\:*?"<>|\x00')
_MAX_NAME_LENGTH = 128


def validate_name(name: str, kind: str = "name") -> str:
    """Validate a single path component and return it unchanged.

    Args:
        name: Raw user input.
        kind: Human-readable label used in the error message.

    Raises:
        InvalidNameError: If the name could escape its parent directory or is
            otherwise unusable as a filesystem path component.
    """
    if not isinstance(name, str) or not name.strip():
        raise InvalidNameError(f"Invalid {kind}: must be a non-empty string")
    if name != name.strip():
        raise InvalidNameError(f"Invalid {kind}: leading/trailing whitespace not allowed")
    if len(name) > _MAX_NAME_LENGTH:
        raise InvalidNameError(f"Invalid {kind}: exceeds {_MAX_NAME_LENGTH} characters")
    if name in (".", "..") or ".." in name:
        raise InvalidNameError(f"Invalid {kind}: '{name}' contains a path traversal sequence")
    if name.startswith("."):
        raise InvalidNameError(f"Invalid {kind}: '{name}' must not start with '.'")
    for ch in name:
        if ch in _FORBIDDEN_NAME_CHARS or ord(ch) < 32:
            raise InvalidNameError(f"Invalid {kind}: '{name}' contains a forbidden character")
    return name


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
            resolved = p.resolve()
            # Manifests are user-writable files; never let a crafted entry turn
            # image serving into an arbitrary local-file read.
            if not (
                resolved.is_relative_to(cls.DATA.resolve())
                or resolved.is_relative_to(cls.PROJECT_ROOT.resolve())
            ):
                raise FileNotFoundError(f"Storage path outside data root: {p}")
            if resolved.exists():
                return resolved
            raise FileNotFoundError(f"Storage path not found: {p}")
        if ".." in p.parts:
            raise FileNotFoundError(f"Storage path contains '..': {p}")
        for candidate in (cls.PROJECT_ROOT / p, cls.DATA / p, Path.cwd() / p):
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
            if (
                resolved.is_relative_to(cls.DATA.resolve())
                or resolved.is_relative_to(cls.PROJECT_ROOT.resolve())
            ):
                return resolved
        raise FileNotFoundError(
            f"Storage path not found: {storage_path}"
        )

    @classmethod
    def try_resolve_storage_path(cls, storage_path: str) -> Optional[Path]:
        """Best-effort variant of :meth:`resolve_storage_path`.

        Listings and exporters iterate over whole manifests, where a single
        stale or hostile record must not abort the operation: unresolvable
        entries simply resolve to ``None`` and are skipped by the caller.
        """
        try:
            return cls.resolve_storage_path(storage_path)
        except (FileNotFoundError, OSError, ValueError):
            return None

    # ── Recipe helpers (standalone, not tied to any single dataset) ──
    @classmethod
    def recipe_dir(cls, name: str) -> Path:
        return cls.RECIPES / validate_name(name, "recipe name")

    @classmethod
    def recipe_yaml_path(cls, name: str) -> Path:
        return cls.recipe_dir(name) / "recipe.yaml"

    # ── Domain Asset helpers ──
    @classmethod
    def dataset_dir(cls, name: str) -> Path:
        return cls.DATASETS / validate_name(name, "dataset name")

    @classmethod
    def dataset_version_dir(cls, dataset_name: str, version: str) -> Path:
        return cls.dataset_dir(dataset_name) / "versions" / validate_name(version, "version")

    @classmethod
    def dataset_manifest_path(cls, dataset_name: str, version: str) -> Path:
        return cls.dataset_version_dir(dataset_name, version) / "manifest.jsonl"

    @classmethod
    def ingestion_run_dir(cls, run_id: str) -> Path:
        return cls.INGESTION_RUNS / validate_name(run_id, "run id")

    # ── Training Asset helpers ──
    @classmethod
    def workspace_dir(cls, name: str) -> Path:
        return cls.WORKSPACE / validate_name(name, "workspace name")

    @classmethod
    def project_dir(cls, name: str) -> Path:
        return cls.PROJECTS / validate_name(name, "project name")

    @classmethod
    def experiment_dir(cls, project_name: str) -> Path:
        return cls.project_dir(project_name) / "experiments"

    @classmethod
    def models_dir(cls, project_name: str) -> Path:
        return cls.project_dir(project_name) / "models"
