"""Changelog — 数据集操作日志（类似 Git commit log）。

记录数据集生命周期中的所有操作，按时间顺序追记。

日志存储：data/datasets/{name}/changelog.jsonl

每行一条 JSON 记录：
  {
    "timestamp": "2026-07-15T10:03:23",
    "action": "dataset.created | version.created | annotation.updated | build.executed | dataset.deleted",
    "detail": "人类可读的描述",
    "user": "操作人",
    "version": "关联版本号",
    "count": 3236       ← 操作涉及的图片/标注数（可选）
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ChangelogEntry:
    """单条操作日志。"""
    timestamp: str = ""
    action: str = ""           # dataset.created / version.created / annotation.updated / build.executed / dataset.deleted
    detail: str = ""
    user: str = ""
    version: str = ""
    count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "action": self.action,
            "detail": self.detail,
            "user": self.user,
            "version": self.version,
            "count": self.count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChangelogEntry":
        return cls(
            timestamp=d.get("timestamp", ""),
            action=d.get("action", ""),
            detail=d.get("detail", ""),
            user=d.get("user", ""),
            version=d.get("version", ""),
            count=d.get("count", 0),
            metadata=d.get("metadata", {}),
        )


class ChangelogStore:
    """操作日志存储（JSONL 文件，追加写入）。"""

    def __init__(self, dataset_dir: Path):
        self.log_path = dataset_dir / "changelog.jsonl"

    def append(self, entry: ChangelogEntry) -> None:
        """追加一条操作日志。"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def list(self, limit: int = 100, offset: int = 0) -> List[ChangelogEntry]:
        """列出操作日志，按时间倒序。"""
        if not self.log_path.exists():
            return []
        entries: List[ChangelogEntry] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(ChangelogEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        # Reverse: newest first
        entries.reverse()
        return entries[offset:offset + limit]

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        count = 0
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    # ── Convenience helpers ──

    def log_dataset_created(self, name: str, task: str = "", user: str = "") -> None:
        self.append(ChangelogEntry(
            action="dataset.created",
            detail=f"Dataset '{name}' created (task: {task})",
            user=user,
        ))

    def log_version_created(self, version: str, detail: str, count: int = 0, user: str = "") -> None:
        self.append(ChangelogEntry(
            action="version.created",
            detail=detail,
            version=version,
            count=count,
            user=user,
        ))

    def log_annotation_updated(self, version: str, image_index: int, box_count: int, user: str = "") -> None:
        self.append(ChangelogEntry(
            action="annotation.updated",
            detail=f"Image {image_index:06d}: {box_count} annotation boxes",
            version=version,
            count=box_count,
            user=user,
        ))

    def log_build_executed(self, version: str, train: int, val: int, user: str = "") -> None:
        self.append(ChangelogEntry(
            action="build.executed",
            detail=f"Train/val split from {version}: {train} train + {val} val images",
            version=version,
            count=train + val,
            user=user,
        ))

    def log_dataset_deleted(self, name: str, user: str = "") -> None:
        self.append(ChangelogEntry(
            action="dataset.deleted",
            detail=f"Dataset '{name}' deleted",
            user=user,
        ))
