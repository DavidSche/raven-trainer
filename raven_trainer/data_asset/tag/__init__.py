"""Tag System — 标签系统。

Tag 不是 Label。Label 是类别（person、helmet），
Tag 是属性（hard_example、occlusion、night、backlight）。

一张图片可以有多个 Tag，同一个 Tag 可以跨类别使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


@dataclass
class TagDef:
    """标签定义。"""
    name: str
    category: str = "general"   # general / quality / scene / lighting / occlusion
    color: str = "#4A90D9"
    description: str = ""


class TagManager:
    """标签管理器——管理标签定义和图片的标签分配。"""

    def __init__(self, asset_dir: Path):
        self.tags_file = asset_dir / "tags.yaml"
        self._tags: Dict[str, TagDef] = {}
        self._load()

    def _load(self) -> None:
        if self.tags_file.exists():
            with open(self.tags_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "tags" in data:
                    for t in data["tags"]:
                        tag = TagDef(**t)
                        self._tags[tag.name] = tag

    def _save(self) -> None:
        data = {
            "tags": [{"name": t.name, "category": t.category, "color": t.color, "description": t.description}
                     for t in self._tags.values()]
        }
        with open(self.tags_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def define(self, name: str, category: str = "general", color: str = "#4A90D9", description: str = "") -> TagDef:
        tag = TagDef(name=name, category=category, color=color, description=description)
        self._tags[name] = tag
        self._save()
        return tag

    def list_tags(self) -> List[TagDef]:
        return list(self._tags.values())

    def get(self, name: str) -> Optional[TagDef]:
        return self._tags.get(name)

    def get_by_category(self, category: str) -> List[TagDef]:
        return [t for t in self._tags.values() if t.category == category]

    def exists(self, name: str) -> bool:
        return name in self._tags

    def undefine(self, name: str) -> None:
        """Remove a tag definition by name."""
        self._tags.pop(name, None)
        self._save()
