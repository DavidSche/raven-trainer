"""Dataset Registry — 数据集注册中心。

类似 Git Repository，管理所有 Dataset 的注册信息。
每一个 Dataset 有唯一的 name 和 asset.yaml。

asset.yaml 结构：
  name: person_detection
  display_name: Workshop Person Dataset
  description: ...
  task: detection
  status: active
  created_at: 2026-07-02T10:00:00
  updated_at: 2026-07-02T10:00:00
  current_version: v3
  version_count: 3
  total_images: 15236
  total_annotations: 18200
  owner: team_vision
  tags: [workshop, person]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

import yaml

from ...config import validate_name


class AssetStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class DataAsset:
    """A registered data asset (dataset).

    每一个 Dataset 在 Registry 中有一个 DataAsset 记录。
    类似 Git 的 repository metadata。
    """

    name: str                                    # 唯一标识符，如 person_detection
    display_name: str = ""                       # 可读名称
    description: str = ""                        # 描述
    task: str = "detection"                       # detection / pose / seg / anomaly
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: str = ""                          # ISO 8601
    updated_at: str = ""                          # ISO 8601
    current_version: str = ""                     # 当前版本号，如 v3
    version_count: int = 0
    total_images: int = 0
    total_annotations: int = 0
    owner: str = ""                               # 负责人
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at or datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "current_version": self.current_version,
            "version_count": self.version_count,
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "owner": self.owner,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataAsset":
        status = AssetStatus(d.get("status", "active"))
        return cls(
            name=d["name"],
            display_name=d.get("display_name", ""),
            description=d.get("description", ""),
            task=d.get("task", "detection"),
            status=status,
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            current_version=d.get("current_version", ""),
            version_count=d.get("version_count", 0),
            total_images=d.get("total_images", 0),
            total_annotations=d.get("total_annotations", 0),
            owner=d.get("owner", ""),
            tags=d.get("tags", []),
            extra=d.get("extra", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path) -> "DataAsset":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class AssetStore:
    """底层存储——管理 data/datasets/ 目录下的所有 Dataset。

    负责：
    - 扫描 data/datasets/ 目录
    - 创建/读取/更新 asset.yaml
    - 管理数据集目录的生命周期
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_assets(self) -> List[DataAsset]:
        """扫描 data/datasets/ 目录，返回所有注册的 DataAsset。"""
        assets: List[DataAsset] = []
        if not self.root.exists():
            return assets
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir():
                asset_file = entry / "asset.yaml"
                if asset_file.exists():
                    try:
                        assets.append(DataAsset.load(asset_file))
                    except Exception as e:
                        print(f"  [WARN] Failed to load asset {entry.name}: {e}")
        return assets

    def get_asset(self, name: str) -> Optional[DataAsset]:
        """根据 name 查找 DataAsset。"""
        asset_file = self.asset_dir(name) / "asset.yaml"
        if asset_file.exists():
            return DataAsset.load(asset_file)
        return None

    def create_asset(self, asset: DataAsset) -> DataAsset:
        """创建新的 DataAsset。"""
        asset.name = validate_name(asset.name, "dataset name")
        asset_dir = self.root / asset.name
        if asset_dir.exists():
            raise FileExistsError(f"Dataset '{asset.name}' already exists at {asset_dir}")
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset.created_at = datetime.now().isoformat()
        asset.updated_at = asset.created_at
        asset_file = asset_dir / "asset.yaml"
        asset.save(asset_file)
        return asset

    def update_asset(self, asset: DataAsset) -> DataAsset:
        """更新已存在的 DataAsset。"""
        asset_file = self.asset_dir(asset.name) / "asset.yaml"
        if not asset_file.exists():
            raise FileNotFoundError(f"Dataset '{asset.name}' not found")
        asset.updated_at = datetime.now().isoformat()
        asset.save(asset_file)
        return asset

    def delete_asset(self, name: str) -> None:
        """删除 DataAsset 及其所有版本和数据目录。"""
        asset_dir = self.asset_dir(name)
        if not asset_dir.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at {asset_dir}")
        shutil.rmtree(asset_dir)

    def asset_dir(self, name: str) -> Path:
        """返回指定 DataAsset 的目录路径。"""
        return self.root / validate_name(name, "dataset name")


class DatasetRegistry:
    """Dataset Registry——数据集注册中心。

    高层 API，供 CLI 和其他模块调用。
    """

    def __init__(self, store: AssetStore):
        self.store = store

    def list(self) -> List[DataAsset]:
        return self.store.list_assets()

    def get(self, name: str) -> Optional[DataAsset]:
        return self.store.get_asset(name)

    def create(self, name: str, task: str = "detection", **kwargs) -> DataAsset:
        asset = DataAsset(name=name, task=task, **kwargs)
        return self.store.create_asset(asset)

    def update(self, asset: DataAsset) -> DataAsset:
        return self.store.update_asset(asset)

    def delete(self, name: str) -> None:
        self.store.delete_asset(name)
