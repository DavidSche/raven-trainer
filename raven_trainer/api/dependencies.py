"""FastAPI dependency injection."""

from __future__ import annotations

from pathlib import Path

from ..config import R
from ..data_asset import DatasetRegistry, AssetStore


def get_registry() -> DatasetRegistry:
    store = AssetStore(R.DATASETS)
    return DatasetRegistry(store)
