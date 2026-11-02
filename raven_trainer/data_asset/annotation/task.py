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

from ...config import R

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
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class TaskManager:
    """标注任务管理器。"""

    def __init__(self, dataset_name: str, version: str = "v1"):
        self.tasks_dir = R.dataset_version_dir(dataset_name, version) / "annotations" / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, task: AnnotationTask) -> Path:
        path = self.tasks_dir / f"task_{task.task_id}.yaml"
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
        path = self.tasks_dir / f"task_{task.task_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Task '{task.task_id}' not found")
        task.save(path)
        return path

    def list_tasks(self) -> List[AnnotationTask]:
        tasks = []
        for f in sorted(self.tasks_dir.glob("task_*.yaml")):
            try:
                tasks.append(AnnotationTask.load(f))
            except Exception:
                pass
        return tasks

    def get_task(self, task_id: str) -> Optional[AnnotationTask]:
        path = self.tasks_dir / f"task_{task_id}.yaml"
        if path.exists():
            return AnnotationTask.load(path)
        return None

    def update_status(self, task_id: str, status: str) -> None:
        if status not in TaskStatus:
            raise ValueError(f"Invalid status: {status}. Must be one of {TaskStatus}")
        task = self.get_task(task_id)
        if task:
            task.status = status
            if status == TaskStatus[2]:  # "completed"
                task.completed_at = datetime.now().isoformat()
            task.save(self.tasks_dir / f"task_{task_id}.yaml")

    def delete_task(self, task_id: str) -> None:
        path = self.tasks_dir / f"task_{task_id}.yaml"
        if path.exists():
            path.unlink()



