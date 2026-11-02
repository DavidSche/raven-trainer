"""Annotation Task — 标注任务生命周期管理。

管理数据集版本与 CVAT/Label Studio 标注任务之间的对应关系。
包括任务创建、导出、重新导入的完整记录。

存储结构：
  data/datasets/<name>/versions/v1/annotations/tasks/
    task_001.yaml     ← CVAT 任务记录
    task_002.yaml     ← Label Studio 任务记录
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ...config import R, validate_name

# Valid task lifecycle statuses
TaskStatus = ("pending", "in_progress", "completed", "failed")


@dataclass
class AnnotationTask:
    """标注任务记录。"""

    task_id: str                         # 在标注平台上的 ID
    platform: str                        # "cvat" | "label_studio"
    name: str                            # 任务名称
    status: str = "pending"              # pending | in_progress | completed | reviewed
    created_at: str = ""
    completed_at: str = ""
    image_count: int = 0
    label_count: int = 0
    export_path: str = ""                # 导入后的标注文件路径
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "platform": self.platform,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at or datetime.now().isoformat(),
            "completed_at": self.completed_at,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "export_path": self.export_path,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "AnnotationTask":
        from dataclasses import MISSING, fields

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Task file must contain a mapping: {path}")
        # Tolerate extra keys written by newer/older versions of the tool.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        required = {
            f.name
            for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - set(filtered)
        if missing:
            raise ValueError(f"Task file {path} is missing required field(s): {sorted(missing)}")
        return cls(**filtered)


class TaskManager:
    """标注任务管理器。"""

    def __init__(self, dataset_name: str, version: str = "v1"):
        self.tasks_dir = R.dataset_version_dir(dataset_name, version) / "annotations" / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        """Resolve a task YAML path, rejecting unsafe ids."""
        return self.tasks_dir / f"task_{validate_name(task_id, 'task id')}.yaml"

    def create_task(self, task: AnnotationTask) -> Path:
        """Create a new task file. Refuses to overwrite an existing task."""
        path = self._task_path(task.task_id)
        if path.exists():
            raise FileExistsError(f"Task '{task.task_id}' already exists")
        task.save(path)
        return path

    def update_task(self, task: AnnotationTask) -> Path:
        """Update an existing task in place by writing back to its YAML file.

        M-35: Previously, callers (e.g. ``update_annotation_task`` in the API
        router) used ``create_task`` to persist edits, which is semantically
        misleading (it implies creation, not update). ``update_task`` requires
        the task to already exist (raising ``FileNotFoundError`` otherwise)
        and writes the updated task object back to the same path, so the
        intent is explicit and missing-task errors surface clearly.
        """
        path = self._task_path(task.task_id)
        if not path.exists():
            raise FileNotFoundError(f"Task '{task.task_id}' not found")
        task.save(path)
        return path

    def list_tasks(self) -> List[AnnotationTask]:
        tasks = []
        for f in sorted(self.tasks_dir.glob("task_*.yaml")):
            try:
                tasks.append(AnnotationTask.load(f))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                # One corrupt task file must not hide the rest.
                continue
        return tasks

    def get_task(self, task_id: str) -> Optional[AnnotationTask]:
        path = self._task_path(task_id)
        if path.exists():
            return AnnotationTask.load(path)
        return None

    def update_status(self, task_id: str, status: str) -> None:
        if status not in TaskStatus:
            raise ValueError(f"Invalid status: {status}. Must be one of {TaskStatus}")
        task = self.get_task(task_id)
        if task:
            task.status = status
            if status == "completed":
                task.completed_at = datetime.now().isoformat()
            task.save(self._task_path(task_id))

    def delete_task(self, task_id: str) -> None:
        path = self._task_path(task_id)
        if path.exists():
            path.unlink()
