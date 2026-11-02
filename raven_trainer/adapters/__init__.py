"""Backend adapter discovery via Python entry_points.

Raven Trainer itself contains NO model backend code.
Each backend (ultralytics, mmdetection, anomalib, etc.) is an independent
pip-installable package that registers itself via entry_points:

    [project.entry-points."raven_trainer.adapters"]
    ultralytics = "raven_ultralytics:RavenUltralyticsAdapter"

Usage:
    pip install raven-trainer          # platform only, no models
    pip install raven-ultralytics       # enables YOLO
    pip install raven-mmdetection       # enables RT-DETR
    pip install raven-anomalib          # enables PatchCore/FastFlow
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Dict, Type

from .base import ModelAdapter, ModelInfo


def discover_adapters() -> Dict[str, Type[ModelAdapter]]:
    """Discover all installed backend adapters.

    1. Scans ``raven_trainer.adapters`` entry point group (pip plugins).
    2. Falls back to builtin adapters (ultralytics, etc.)
    """
    registry: Dict[str, Type[ModelAdapter]] = {}
    
    # Step 1: entry_points discovery
    eps = entry_points(group="raven_trainer.adapters")
    for ep in eps:
        try:
            cls = ep.load()
            if hasattr(cls, "backend_name"):
                registry[cls.backend_name] = cls
            else:
                registry[ep.name] = cls
        except Exception:
            pass
    
    # Step 2: builtin adapters (no entry_points needed)
    try:
        from .builtin import BUILTIN_ADAPTERS
        for name, cls in BUILTIN_ADAPTERS.items():
            if name not in registry:
                registry[name] = cls
    except ImportError:
        pass
    
    return registry


def load_adapter(name: str) -> Type[ModelAdapter]:
    """Load adapter class by backend name.

    Discovers from installed packages. If not found,
    suggests the corresponding pip package.
    """
    adapters = discover_adapters()
    if name not in adapters:
        suggestions = {
            "ultralytics": "pip install raven-ultralytics",
            "mmdetection": "pip install raven-mmdetection",
            "anomalib": "pip install raven-anomalib",
            "rfdetr": "pip install raven-rfdetr",
        }
        hint = suggestions.get(name, f"pip install raven-{name}")
        available = list(adapters.keys())
        raise KeyError(
            f"Backend '{name}' not found. {hint}\n"
            f"Installed backends: {available}"
        )
    return adapters[name]


def list_adapters() -> Dict[str, ModelInfo]:
    """List all installed adapters and their metadata."""
    return {name: cls.info() for name, cls in discover_adapters().items()}


__all__ = [
    "ModelAdapter",
    "ModelInfo",
    "load_adapter",
    "list_adapters",
    "discover_adapters",
]
