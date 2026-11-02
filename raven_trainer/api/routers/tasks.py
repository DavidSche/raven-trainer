"""Tasks API — 标注任务管理。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas import AnnotationTaskSchema
from ...data_asset.annotation.task import TaskManager, AnnotationTask, TaskStatus

router = APIRouter(tags=["tasks"])


@router.get("/datasets/{dataset_name}/tasks")
def list_annotation_tasks(
    dataset_name: str,
    version: str = Query("v1"),
    platform: Optional[str] = Query(None),
):
    tm = TaskManager(dataset_name, version)
    tasks = tm.list_tasks()
    if platform:
        tasks = [t for t in tasks if t.platform == platform]
    return [AnnotationTaskSchema(
        task_id=t.task_id,
        platform=t.platform,
        name=t.name,
        status=t.status,
        image_count=t.image_count,
        label_count=t.label_count,
        created_at=t.created_at,
        completed_at=t.completed_at,
        notes=t.notes,
    ) for t in tasks]


@router.post("/datasets/{dataset_name}/tasks")
def create_annotation_task(dataset_name: str, task: AnnotationTaskSchema, version: str = Query("v1")):
    tm = TaskManager(dataset_name, version)
    new_task = AnnotationTask(
        task_id=task.task_id,
        platform=task.platform,
        name=task.name,
        status="pending",
        image_count=task.image_count,
    )
    try:
        tm.create_task(new_task)
    except FileExistsError as exc:
        raise HTTPException(409, f"Task '{task.task_id}' already exists") from exc
    return {"status": "created", "task_id": task.task_id}


@router.get("/datasets/{dataset_name}/tasks/{task_id}")
def get_annotation_task(dataset_name: str, task_id: str, version: str = Query("v1")):
    tm = TaskManager(dataset_name, version)
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return AnnotationTaskSchema(
        task_id=task.task_id,
        platform=task.platform,
        name=task.name,
        status=task.status,
        image_count=task.image_count,
        label_count=task.label_count,
        created_at=task.created_at,
        completed_at=task.completed_at,
        notes=task.notes,
    )


@router.put("/datasets/{dataset_name}/tasks/{task_id}")
def update_annotation_task(dataset_name: str, task_id: str, data: dict, version: str = Query("v1")):
    tm = TaskManager(dataset_name, version)
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task '{task_id}' not found")
    if "status" in data:
        new_status = data["status"]
        if new_status not in TaskStatus:
            raise HTTPException(
                400, f"Invalid status '{new_status}'. Must be one of: {list(TaskStatus)}"
            )
        task.status = new_status
        # M-35: Use the dedicated update_task method instead of create_task,
        # which previously silently overwrote the task file and masked
        # "task missing" errors. update_task raises FileNotFoundError if
        # the task was deleted between get_task and the write.
        try:
            tm.update_task(task)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Task '{task_id}' not found") from exc
    return {"status": "updated"}
