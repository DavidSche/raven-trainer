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

from ...config import validate_name
from ..constants import IMAGE_EXTS


def _version_sort_key(name: str) -> tuple:
    """Sort key that orders ``v2`` before ``v10``.

    Plain string sorting puts ``v10`` before ``v9``, which silently made
    ``list_versions()[0]`` ("the latest version") wrong from v10 onwards.
    Auto-numbered versions sort numerically; anything else sorts after them
    by name so hand-crafted tags stay visible.
    """
    if name.startswith("v") and name[1:].isdigit():
        return (0, int(name[1:]), name)
    return (1, 0, name)


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
        """列出所有版本，按版本号降序排列（v10 排在 v9 之前）。"""
        snapshots: List[VersionSnapshot] = []
        if not self.versions_dir.exists():
            return snapshots
        entries = [e for e in self.versions_dir.iterdir() if e.is_dir()]
        for entry in sorted(entries, key=lambda e: _version_sort_key(e.name), reverse=True):
            snap_file = entry / "snapshot.yaml"
            if snap_file.exists():
                try:
                    snapshots.append(VersionSnapshot.load(snap_file))
                except (OSError, yaml.YAMLError, KeyError, TypeError):
                    # A single unreadable snapshot must not hide the others.
                    continue
        return snapshots

    def get_version(self, version: str) -> Optional[VersionSnapshot]:
        """获取指定版本的快照。"""
        snap_file = self.get_version_dir(version) / "snapshot.yaml"
        if snap_file.exists():
            return VersionSnapshot.load(snap_file)
        return None

    def next_version_str(self) -> str:
        """Return the next auto-numbered version (``max(vN) + 1``).

        Uses the maximum existing number rather than ``len(versions) + 1`` so
        that deleting an old version can never make a later create overwrite
        an existing, immutable snapshot.
        """
        max_num = 0
        for v in self.list_versions():
            num_str = v.version[1:] if v.version.startswith("v") and v.version[1:].isdigit() else ""
            if num_str and int(num_str) > max_num:
                max_num = int(num_str)
        return f"v{max_num + 1}"

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
        """创建新版本（append-only）。

        版本目录先写入临时目录，全部写成功后才原子重命名为最终目录，
        避免半途失败留下一个“看起来存在但内容不全”的版本。
        Pinned by tests: ``create_version`` never overwrites an existing tag.
        """
        existing = self.list_versions()
        # Lineage always points at the newest existing snapshot, also when the
        # caller pins an explicit tag (a manual tag is still a successor).
        parent = existing[0].version if existing else None

        if version_tag:
            version_str = validate_name(version_tag, "version")
            if (self.versions_dir / version_str).exists():
                raise FileExistsError(f"Version '{version_str}' already exists")
        else:
            version_str = self.next_version_str()

        final_dir = self.versions_dir / version_str
        if final_dir.exists():
            raise FileExistsError(f"Version '{version_str}' already exists")
        # Hidden temp dir: ignored by list_versions() and cleaned up on failure.
        tmp_dir = self.versions_dir / f".tmp-{version_str}-{datetime.now():%Y%m%d%H%M%S%f}"

        image_count = 0
        try:
            annotations_dir = tmp_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=False)
            (annotations_dir / "raw").mkdir(exist_ok=True)
            (annotations_dir / "converted").mkdir(exist_ok=True)

            if manifest_records:
                from ...asset_manager import AssetManager

                am = AssetManager()
                records = []
                for rec in manifest_records:
                    record = am.create_manifest_record(**rec) if isinstance(rec, dict) else rec
                    records.append(record)
                am.save_manifest(records, tmp_dir / "manifest.jsonl")
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
            # snapshot.yaml is written last: its presence is what makes the
            # version visible to list_versions().
            snapshot.save(tmp_dir / "snapshot.yaml")
            tmp_dir.rename(final_dir)
            return snapshot
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    def get_version_dir(self, version: str) -> Path:
        """获取指定版本的目录。"""
        return self.versions_dir / validate_name(version, "version")
