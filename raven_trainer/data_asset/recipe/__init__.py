"""Dataset Recipe — 构建规则，数据集的可复现"源代码"。

Recipe 定义如何从原始数据构建训练集：
  - 使用哪些 Sources
  - 如何过滤（include/exclude）
  - 采样策略（motion, scene_change）
  - 分割策略（by_video, random）
  - 增强策略
  - 均衡策略（camera, time）

保存 Recipe，不保存训练集。任何时候都可以重新 Build。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RecipeSource:
    name: str
    path: Optional[str] = None
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)


@dataclass
class RecipeSplit:
    strategy: str = "by_video"
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1


@dataclass
class RecipeAugment:
    copy_paste: bool = True
    hsv: bool = True
    motion_blur: bool = False
    fliplr: float = 0.5


@dataclass
class DatasetRecipe:
    name: str
    version: str = "1.0"
    description: str = ""
    created_at: str = ""
    sources: List[RecipeSource] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    sampling: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "motion,scene_change",
        "motion_threshold": 0.4,
        "scene_threshold": 0.3,
    })
    split: RecipeSplit = field(default_factory=RecipeSplit)
    augment: RecipeAugment = field(default_factory=RecipeAugment)
    balance: Dict[str, Any] = field(default_factory=lambda: {
        "max_per_camera": 5000,
        "time_balance": True,
    })
    quality: Dict[str, Any] = field(default_factory=lambda: {
        "min_blur_score": 100.0,
        "dedup_method": "dhash",
    })
    build_target: str = "yolo"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetRecipe":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRecipe":
        sources = [RecipeSource(**s) for s in data.get("sources", [])]
        split_data = data.get("split", {})
        split = RecipeSplit(**split_data) if split_data else RecipeSplit()
        augment_data = data.get("augment", {})
        augment = RecipeAugment(**augment_data) if augment_data else RecipeAugment()
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            sources=sources,
            include=data.get("include", []),
            exclude=data.get("exclude", []),
            sampling=data.get("sampling", {}),
            split=split,
            augment=augment,
            balance=data.get("balance", {}),
            quality=data.get("quality", {}),
            build_target=data.get("build_target", "yolo"),
            extra=data.get("extra", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at or datetime.now().isoformat(),
            "sources": [{"name": s.name, "path": s.path, "include": s.include, "exclude": s.exclude}
                        for s in self.sources],
            "include": self.include,
            "exclude": self.exclude,
            "sampling": self.sampling,
            "split": {"strategy": self.split.strategy, "train": self.split.train, "val": self.split.val, "test": self.split.test},
            "augment": {"copy_paste": self.augment.copy_paste, "hsv": self.augment.hsv, "motion_blur": self.augment.motion_blur, "fliplr": self.augment.fliplr},
            "balance": self.balance,
            "quality": self.quality,
            "build_target": self.build_target,
            "extra": self.extra,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def save_to_asset(self, asset_dir: Path, recipe_name: str = "default") -> Path:
        recipe_dir = asset_dir / "recipes"
        recipe_dir.mkdir(parents=True, exist_ok=True)
        path = recipe_dir / f"recipe_{validate_name(recipe_name, 'recipe name')}.yaml"
        self.save(path)
        return path


class RecipeManager:
    """Recipe 管理器——管理一个 Dataset 的所有 Recipe。"""

    def __init__(self, asset_dir: Path):
        self.recipes_dir = asset_dir / "recipes"
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

    def list_recipes(self) -> List[Path]:
        if not self.recipes_dir.exists():
            return []
        return sorted(self.recipes_dir.glob("recipe_*.yaml"))

    def get_recipe(self, name: str = "default") -> Optional[DatasetRecipe]:
        path = self.recipes_dir / f"recipe_{validate_name(name, 'recipe name')}.yaml"
        if path.exists():
            return DatasetRecipe.from_yaml(path)
        return None

    def save_recipe(self, recipe: DatasetRecipe, name: str = "default") -> Path:
        return recipe.save_to_asset(self.recipes_dir.parent, name)

# ──────────────────────────────────────────────
#  Global Recipe Catalog（独立存储，不依赖单个 Dataset）
#  data/recipes/<name>/recipe.yaml
# ──────────────────────────────────────────────


from ...config import R as _R, validate_name

class CatalogRecipeManager:
    """全局 Recipe 目录——独立于任何单个数据集。

    Recipe 是资产数据集和训练工作区之间的多对多桥梁。
    一个 Recipe 可以引用多个资产数据集，一个资产数据集可以被多个 Recipe 引用。

    存储: data/recipes/<name>/recipe.yaml
    """

    def __init__(self, recipes_root: Optional[Path] = None):
        self.recipes_root = recipes_root or _R.RECIPES
        self.recipes_root.mkdir(parents=True, exist_ok=True)

    def list_recipes(self) -> List[str]:
        """返回所有 Recipe 的名称列表。"""
        if not self.recipes_root.exists():
            return []
        return sorted(
            d.name for d in self.recipes_root.iterdir()
            if d.is_dir() and (d / "recipe.yaml").exists()
        )

    def get_recipe(self, name: str) -> Optional[DatasetRecipe]:
        """按名称获取 Recipe。"""
        path = self.recipes_root / validate_name(name, "recipe name") / "recipe.yaml"
        if path.exists():
            return DatasetRecipe.from_yaml(path)
        return None

    def save_recipe(self, recipe: DatasetRecipe, name: Optional[str] = None) -> Path:
        """保存 Recipe。如果未指定名称则使用 recipe.name。"""
        recipe_name = validate_name(name or recipe.name, "recipe name")
        path = self.recipes_root / recipe_name / "recipe.yaml"
        recipe.save(path)
        return path

    def delete_recipe(self, name: str) -> None:
        """删除 Recipe。"""
        import shutil
        path = self.recipes_root / validate_name(name, "recipe name")
        if path.exists():
            shutil.rmtree(path)

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """列出可用的资产数据集（用于 UI 选择）。"""
        from ..registry import AssetStore
        store = AssetStore(_R.DATASETS)
        assets = store.list_assets()
        result = []
        for a in assets:
            from ..version import VersionManager
            vm = VersionManager(_R.DATASETS / a.name)
            versions = vm.list_versions()
            result.append({
                "name": a.name,
                "display_name": a.display_name or a.name,
                "task": a.task,
                "current_version": a.current_version,
                "version_count": a.version_count,
                "total_images": a.total_images,
                "versions": [v.version for v in versions],
            })
        return result
