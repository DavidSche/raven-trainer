"""Shared CLI helpers for locating model artifacts.

P0-3: rf-detr checkpoints are ``.pth`` while YOLO uses ``.pt`` — the three
CLIs (train/export/eval) must share one weight locator so a fourth caller
cannot regress again.  ``v*.pt`` + ``v*.pth`` globs, lexicographically largest
wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def list_model_versions(models_dir: Path) -> List[Path]:
    """All versioned model weights (``v*.pt`` / ``v*.pth``), sorted ascending."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    files = sorted(models_dir.glob("v*.pt")) + sorted(models_dir.glob("v*.pth"))
    return sorted(files, key=lambda p: p.name)


def latest_model_path(models_dir: Path) -> Optional[Path]:
    """Latest versioned model weight (``v*.pt`` + ``v*.pth``), or None."""
    files = list_model_versions(models_dir)
    return files[-1] if files else None


def next_model_path(models_dir: Path, version: Optional[str] = None) -> Path:
    """Compute the next versioned model path (``v1.1.pt`` style) in models_dir.

    Without ``version``, increments the latest existing version (e.g. v1.0 ->
    v1.1); falls back to v1.0.  With ``version``, returns
    ``v{version}.{ext}`` where ext is derived from the latest weight's suffix
    (``.pt`` for YOLO, ``.pth`` for rf-detr), defaulting to ``.pt``.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    latest = latest_model_path(models_dir)
    ext = latest.suffix if latest else ".pt"

    if version:
        return models_dir / f"v{version}{ext}"

    if latest:
        # v1.0 -> v1.1
        stem = latest.stem  # e.g. v1.0
        try:
            base = stem[1:]  # strip leading v
            parts = base.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return models_dir / f"v{'.'.join(parts)}{ext}"
        except (ValueError, IndexError):
            pass
    return models_dir / f"v1.0{ext}"
