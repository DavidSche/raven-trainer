"""Misdata / Feedback API — 误报/反馈管理。"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query

from ...config import R
from ...data_asset.feedback import FEEDBACK_CATEGORIES, FeedbackManager

router = APIRouter(tags=["feedback"])

# Safe filename pattern: letters, digits, underscore, hyphen, dot. No path separators or ..
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _validate_category(category: str) -> None:
    """Reject categories that aren't in the allowed set or contain path tricks."""
    if category not in FEEDBACK_CATEGORIES:
        raise HTTPException(400, f"Invalid category: {category!r}")


def _validate_filename(filename: str) -> None:
    """Reject filenames that could enable path traversal."""
    if not filename:
        raise HTTPException(400, "filename required")
    if not _SAFE_FILENAME_RE.match(filename) or ".." in filename:
        raise HTTPException(400, "Invalid filename: only letters, digits, '_', '-', '.' are allowed")


def _get_feedback_manager(dataset_name: str) -> FeedbackManager:
    return FeedbackManager(dataset_name)


def _get_thumbnail(path: str, max_size: int = 320) -> str | None:
    """Generate a base64 thumbnail of an image."""
    try:
        from PIL import Image
        import io
        img_path = Path(path)
        if not img_path.exists():
            return None
        # Use a context manager so the file handle is released even on error
        with Image.open(img_path) as img:
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


@router.get("/feedback/datasets")
async def list_feedback_datasets():
    """List datasets that have feedback data."""
    results = []
    datasets_dir = R.DATASETS
    if datasets_dir.exists():
        for ds_dir in sorted(datasets_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            feedback_dir = ds_dir / "feedback"
            if feedback_dir.exists() and any(
                (feedback_dir / cat).exists() and len(list((feedback_dir / cat).iterdir())) > 0
                for cat in ["false_positive", "false_negative", "hard_case", "customer_case"]
            ):
                results.append(ds_dir.name)
    return results


@router.get("/feedback/{dataset_name}")
async def list_feedback(
    dataset_name: str,
    category: str = Query(None),
    include_thumbnail: bool = Query(False),
):
    """List feedback items for a dataset, optionally with thumbnails."""
    fm = _get_feedback_manager(dataset_name)
    items = fm.list_feedback(category)

    if include_thumbnail:
        for item in items:
            thumb = _get_thumbnail(item["path"])
            if thumb:
                item["thumbnail"] = thumb

    count_by_status = {"pending": 0, "approved": 0, "rejected": 0}
    for item in items:
        s = item.get("status", "pending")
        count_by_status[s] = count_by_status.get(s, 0) + 1

    return {
        "items": items,
        "summary": count_by_status,
        "total": len(items),
    }


@router.get("/feedback/{dataset_name}/categories")
async def feedback_category_summary(dataset_name: str):
    """Get category breakdown for a dataset."""
    from ...data_asset.feedback import FEEDBACK_CATEGORIES
    fm = _get_feedback_manager(dataset_name)
    result = {}
    for cat in FEEDBACK_CATEGORIES:
        items = fm.list_feedback(cat)
        result[cat] = {
            "total": len(items),
            "pending": sum(1 for i in items if i["status"] == "pending"),
            "approved": sum(1 for i in items if i["status"] == "approved"),
            "rejected": sum(1 for i in items if i["status"] == "rejected"),
        }
    return result


@router.post("/feedback/{dataset_name}/approve")
async def approve_feedback(dataset_name: str, data: dict = Body(default_factory=dict)):
    category = data.get("category", "false_positive")
    filename = data.get("filename", "")
    _validate_category(category)
    _validate_filename(filename)
    fm = _get_feedback_manager(dataset_name)
    fm.approve(category, filename)
    return {"status": "approved", "category": category, "filename": filename}


@router.post("/feedback/{dataset_name}/reject")
async def reject_feedback(dataset_name: str, data: dict = Body(default_factory=dict)):
    category = data.get("category", "false_positive")
    filename = data.get("filename", "")
    _validate_category(category)
    _validate_filename(filename)
    fm = _get_feedback_manager(dataset_name)
    fm.reject(category, filename)
    return {"status": "rejected", "category": category, "filename": filename}


@router.post("/feedback/{dataset_name}/merge")
async def merge_feedback(dataset_name: str, data: dict = Body(default_factory=dict)):
    fm = _get_feedback_manager(dataset_name)
    version = data.get("version", "")
    count = fm.merge(version)
    return {"status": "merged", "count": count}


@router.get("/feedback/{dataset_name}/thumbnail/{filename:path}")
async def feedback_thumbnail(dataset_name: str, filename: str, category: str = Query("false_positive")):
    from fastapi.responses import FileResponse
    _validate_category(category)
    _validate_filename(filename)
    fm = _get_feedback_manager(dataset_name)
    cat_dir = fm.feedback_dir / category
    file_path = (cat_dir / filename).resolve()
    # Defense-in-depth: ensure resolved path is still under the category dir
    try:
        file_path.relative_to(cat_dir.resolve())
    except ValueError:
        raise HTTPException(400, "Path traversal detected")
    if file_path.exists():
        return FileResponse(str(file_path))
    raise HTTPException(404, "Thumbnail not found")
