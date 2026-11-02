"""Dataset Version — 不可变版本管理。

每个 Version 是一个不可修改的快照（Append-Only）。
任何修改都生成新版本，绝不覆盖旧版本。

存储结构：
  data/assets/<name>/versions/
    v1/
      snapshot.yaml     ← 版本快照元数据
      images/           ← 该版本的图片（或符号链接）
      annotations/      ← 该版本的标注
    v2/
      ...
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..constants import IMAGE_EXTS


@dataclass
class VersionSnapshot:
    """单个版本快照的元数据。

    每个版本是 append-only 的，创建后不可修改。
    """

    version: str                             # v1, v2, v3...
    parent_version: Optional[str] = None     # 上一版本的版本号
    created_at: str = ""
    description: str = ""
    image_count: int = 0
    annotation_count: int = 0
    source: str = ""                         # 来源描述，如 "ingestion run at 2026-07-02"
    recipe_ref: str = ""                     # 引用的 recipe，如 "recipe_default"
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at or datetime.now().isoformat(),
            "description": self.description,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "source": self.source,
            "recipe_ref": self.recipe_ref,
            "tags": self.tags,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VersionSnapshot":
        return cls(
            version=d["version"],
            parent_version=d.get("parent_version"),
            created_at=d.get("created_at", ""),
            description=d.get("description", ""),
            image_count=d.get("image_count", 0),
            annotation_count=d.get("annotation_count", 0),
            source=d.get("source", ""),
            recipe_ref=d.get("recipe_ref", ""),
            tags=d.get("tags", []),
            created_by=d.get("created_by", ""),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path) -> "VersionSnapshot":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class VersionManager:
    """版本管理器——管理单个 Dataset 的所有版本。

    类似 Git 的版本管理，但简化了分支概念（v1.0 只支持线性版本）。
    """

    def __init__(self, asset_dir: Path):
        self.versions_dir = asset_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> List[VersionSnapshot]:
        """列出所有版本，按版本号降序排列。"""
        snapshots: List[VersionSnapshot] = []
        if not self.versions_dir.exists():
            return snapshots
        for entry in sorted(self.versions_dir.iterdir(), reverse=True):
            if entry.is_dir():
                snap_file = entry / "snapshot.yaml"
                if snap_file.exists():
                    try:
                        snapshots.append(VersionSnapshot.load(snap_file))
                    except Exception:
                        pass
        return snapshots

    def get_version(self, version: str) -> Optional[VersionSnapshot]:
        """获取指定版本的快照。"""
        snap_file = self.versions_dir / version / "snapshot.yaml"
        if snap_file.exists():
            return VersionSnapshot.load(snap_file)
        return None

    def create_version(
        self,
        manifest_records=None,
        source_images=None,
        description="",
        recipe_ref="",
        tags=None,
        asset_manager=None,
        source="manual",
        version_tag="",
    ) -> "VersionSnapshot":
        # Use user-specified version_tag if provided, otherwise auto-number
        if version_tag:
            version_str = version_tag
            parent = None
            existing = self.list_versions()
        else:
            existing = self.list_versions()
            next_num = 1
            parent = None
            # Use max(version_number) + 1 across all existing versions rather
            # than relying solely on existing[0] (which is sorted but may skip
            # deleted versions) or len(existing)+1 (which collides after deletes).
            if existing:
                max_num = 0
                for v in existing:
                    try:
                        # Only strip a single leading 'v' (lstrip('v') would
                        # eat 'vv1' -> '1' too, masking naming violations).
                        num_str = v.version[1:] if v.version.startswith("v") and v.version[1:].isdigit() else ""
                        if num_str:
                            n = int(num_str)
                            if n > max_num:
                                max_num = n
                    except (ValueError, AttributeError):
                        continue
                next_num = max_num + 1
                parent = existing[0].version
            version_str = f"v{next_num}"
        version_dir = self.versions_dir / version_str
        annotations_dir = version_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        (annotations_dir / "raw").mkdir(exist_ok=True)
        (annotations_dir / "converted").mkdir(exist_ok=True)

        image_count = 0

        if manifest_records:
            import json
            from ...asset_manager import AssetManager
            am = AssetManager()
            records = []
            for rec in manifest_records:
                record = am.create_manifest_record(**rec) if isinstance(rec, dict) else rec
                records.append(record)
            am.save_manifest(records, version_dir / "manifest.jsonl")
            image_count = len(records)
        elif source_images and source_images.exists():
            image_count = sum(1 for f in source_images.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS)

        snapshot = VersionSnapshot(
            version=version_str,
            parent_version=parent,
            description=description,
            image_count=image_count,
            source=source or (f"created from {source_images}" if source_images else "manual"),
            recipe_ref=recipe_ref,
            tags=tags or [],
        )
        snapshot.save(version_dir / "snapshot.yaml")
        return snapshot
    def get_version_dir(self, version: str) -> Path:
        """获取指定版本的目录。"""
        return self.versions_dir / version

