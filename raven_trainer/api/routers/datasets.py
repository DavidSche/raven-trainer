"""Datasets API — 数据集管理。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path as FPath, Query
from fastapi.responses import FileResponse

from ..schemas import (
    DatasetSchema, VersionSchema, TagSchema,
    AnnotationStatsSchema, AnnotationTaskSchema,
    IntelligenceReportSchema, HealthReportSchema,
    MisdataItemSchema, LineageSchema, LineageNodeSchema,
)
from ..dependencies import get_registry
from ...config import R
from pydantic import BaseModel, Field
from ...data_asset import (
    AssetStore, DatasetRegistry, VersionManager, TagManager,
    DatasetIntelligence, DatasetDoctor,
    FeedbackManager,
    DatasetRegistry as Registry,
    AnnotationTask, TaskManager,
    ChangelogStore, ChangelogEntry,
)

router = APIRouter(tags=["datasets"])


# ── List / Create / Delete ──

@router.get("/datasets")
async def list_datasets(registry: DatasetRegistry = Depends(get_registry)):
    assets = registry.list()
    return [DatasetSchema(
        name=a.name,
        display_name=a.display_name or a.name,
        task=a.task,
        status=a.status.value if hasattr(a.status, 'value') else a.status,
        current_version=a.current_version,
        version_count=a.version_count,
        total_images=a.total_images,
        total_annotations=a.total_annotations,
        health_score=0.0,
        created_at=a.created_at,
        updated_at=a.updated_at,
        tags=a.tags,
        owner=a.owner,
    ) for a in assets]


@router.get("/datasets/{name}")
async def get_dataset(name: str, registry: DatasetRegistry = Depends(get_registry)):
    asset = registry.get(name)
    if not asset:
        raise HTTPException(404, f"Dataset '{name}' not found")
    return DatasetSchema(
        name=asset.name,
        display_name=asset.display_name or asset.name,
        task=asset.task,
        status=asset.status.value if hasattr(asset.status, 'value') else asset.status,
        current_version=asset.current_version,
        version_count=asset.version_count,
        total_images=asset.total_images,
        total_annotations=asset.total_annotations,
        health_score=0.0,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
        tags=asset.tags,
        owner=asset.owner,
    )


# ── Versions ──

@router.get("/datasets/{name}/versions")
async def list_versions(name: str):
    vm = VersionManager(R.dataset_dir(name))
    versions = vm.list_versions()
    return [VersionSchema(
        version=v.version,
        image_count=v.image_count,
        annotation_count=v.annotation_count,
        source=v.source,
        created_at=v.created_at,
        parent_version=v.parent_version,
    ) for v in versions]


# ── Intelligence ──

@router.get("/datasets/{name}/analyze")
async def analyze_dataset(name: str, version: str = Query("v1")):
    di = DatasetIntelligence(name, version)
    report = di.analyze()
    return IntelligenceReportSchema(
        dataset_name=report.dataset_name,
        version=report.version,
        created_at=report.created_at,
        total_images=report.total_images,
        total_annotations=report.total_annotations,
        annotated_images=report.annotated_images,
        background_images=report.background_images,
        background_ratio=report.background_ratio,
        class_distribution=report.class_distribution,
        camera_distribution=report.camera_distribution,
        camera_bias_score=report.camera_bias_score,
        time_distribution=report.time_distribution,
        object_size_bins=report.object_size_bins,
        small_object_ratio=report.small_object_ratio,
        medium_object_ratio=report.medium_object_ratio,
        large_object_ratio=report.large_object_ratio,
        bias_warnings=report.bias_warnings,
        recommendations=report.recommendations,
    )


# ── Doctor (Health Check) ──

@router.get("/datasets/{name}/doctor")
async def doctor_dataset(name: str, version: str = Query("v1")):
    doctor = DatasetDoctor(R.dataset_dir(name))
    # Gather dataset stats for health check
    import json as _json
    manifest_path = R.dataset_manifest_path(name, version)
    metadata_count = 0
    source_distribution = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata_count += 1
                    try:
                        rec = _json.loads(line)
                        src = rec.get("source", "unknown")
                        source_distribution[src] = source_distribution.get(src, 0) + 1
                    except _json.JSONDecodeError:
                        pass
    ann_dir = R.dataset_version_dir(name, version) / "annotations" / "converted" / "yolo"
    report = doctor.check(
        metadata_count=metadata_count,
        source_distribution=source_distribution,
        manifest_path=manifest_path,
        ann_dir=ann_dir if ann_dir.exists() else None,
    )
    return HealthReportSchema(
        dataset_name=name,
        version=version,
        created_at=report.created_at,
        overall_score=report.overall_score,
        checks=[{'name': c.name, 'status': c.status, 'score': c.score, 'message': c.message, 'detail': c.detail} for c in report.checks],
        issues=[c.message for c in report.checks if c.status != 'pass'],
        summary=report.summary,
        recommendations=report.recommendations,
    )


# ── Tags ──

@router.get("/datasets/{name}/tags")
async def list_tags(name: str):
    tm = TagManager(R.dataset_dir(name))
    tags = tm.list_tags()
    return [TagSchema(name=t.name, color=t.color, description=t.description) for t in tags]


@router.post("/datasets/{name}/tags")
async def add_tag(name: str, tag: TagSchema):
    tm = TagManager(R.dataset_dir(name))
    tm.define(tag.name, color=tag.color, description=tag.description)
    return {"status": "ok"}


# ── Misdata / Feedback ──

@router.get("/datasets/{name}/misdata")
async def list_misdata(name: str, category: Optional[str] = Query(None)):
    fm = FeedbackManager(name)
    items = fm.list_feedback(category)
    return [MisdataItemSchema(**item) for item in items]


@router.post("/datasets/{name}/misdata/approve")
async def approve_misdata(name: str, data: dict):
    fm = FeedbackManager(name)
    fm.approve(data.get("category", "false_positive"), data["filename"])
    return {"status": "ok"}


@router.post("/datasets/{name}/misdata/reject")
async def reject_misdata(name: str, data: dict):
    fm = FeedbackManager(name)
    fm.reject(data.get("category", "false_positive"), data["filename"])
    return {"status": "ok"}


# ── Lineage ──

@router.get("/datasets/{name}/lineage")
async def get_lineage(name: str):
    from ...data_asset.lineage import LineageStore
    store = LineageStore(R.dataset_dir(name))
    lineage = store.get_lineage()
    return LineageSchema(
        nodes=[LineageNodeSchema(id=n.id, type=n.type, label=n.label, created_at=n.created_at) for n in lineage.nodes],
        edges=[{"from": e.source, "to": e.target, "type": e.type} for e in lineage.edges],
    )


# ── Annotation Tasks ──

@router.get("/datasets/{name}/tasks")
async def list_tasks(name: str, version: str = Query("v1")):
    tm = TaskManager(name, version)
    tasks = tm.list_tasks()
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




# ── Create / Delete ──

class CreateDatasetRequest(BaseModel):
    name: str
    display_name: str = ""
    task: str = "detection"
    owner: str = ""


@router.post("/datasets")
async def create_dataset(req: CreateDatasetRequest):
    """Create a new dataset asset."""
    store = AssetStore(R.DATASETS)
    registry = DatasetRegistry(store)
    asset = registry.create(
        name=req.name,
        display_name=req.display_name or req.name,
        task=req.task,
        owner=req.owner,
    )
    ChangelogStore(R.dataset_dir(req.name)).log_dataset_created(req.name, req.task)
    return {"status": "created", "name": asset.name}


@router.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Delete a dataset asset and all its versions."""
    store = AssetStore(R.DATASETS)
    registry = DatasetRegistry(store)
    registry.delete(name)
    ChangelogStore(R.dataset_dir(name)).log_dataset_deleted(name)
    return {"status": "deleted", "name": name}


# ── Versions ──

class CreateVersionRequest(BaseModel):
    source: str = "manual"
    tags: list[str] = []
    notes: str = ""
    version: str = ""  # optional version tag like "1.0.0"; auto v1/v2 if empty


@router.post("/datasets/{name}/versions")
async def create_version(name: str, req: CreateVersionRequest):
    """Create a new dataset version."""
    vm = VersionManager(R.dataset_dir(name))
    store = AssetStore(R.DATASETS)
    registry = DatasetRegistry(store)
    asset = registry.get(name)
    if not asset:
        raise HTTPException(404, f"Dataset '{name}' not found")
    version = vm.create_version(source=req.source, tags=req.tags, version_tag=req.version)
    # Update registry metadata
    asset.current_version = version.version
    asset.version_count = len(vm.list_versions())
    registry.update(asset)
    ChangelogStore(R.dataset_dir(name)).log_version_created(version.version, f"Manual version creation ({req.source})")

# ── Recipe CRUD ──

@router.get("/datasets/{name}/recipes")
async def list_recipes(name: str):
    """List all recipe names for a dataset."""
    from ...data_asset.recipe import RecipeManager
    rm = RecipeManager(R.dataset_dir(name))
    recipes = rm.list_recipes()
    return [rp.stem.replace("recipe_", "") for rp in recipes]


@router.get("/datasets/{name}/recipes/{recipe_name}")
async def get_recipe(name: str, recipe_name: str):
    """Get a recipe by name."""
    from ...data_asset.recipe import RecipeManager
    rm = RecipeManager(R.dataset_dir(name))
    recipe = rm.get_recipe(recipe_name)
    if recipe is None:
        raise HTTPException(404, f"Recipe '{recipe_name}' not found")
    return recipe.to_dict()


@router.post("/datasets/{name}/recipes/default")
async def create_default_recipe(name: str):
    """Create a default recipe for a dataset."""
    from ...data_asset.recipe import DatasetRecipe, RecipeManager
    rm = RecipeManager(R.dataset_dir(name))
    existing = rm.get_recipe("default")
    if existing:
        return {"status": "exists", "path": str(rm.recipes_dir / "recipe_default.yaml")}
    recipe = DatasetRecipe(
        name=name,
        description=f"Default recipe for {name}",
    )
    path = rm.save_recipe(recipe)
    return {"status": "created", "path": str(path)}


@router.post("/datasets/{name}/recipes/{recipe_name}")
async def save_recipe(name: str, recipe_name: str, recipe_data: dict):
    """Save a recipe (create or update)."""
    from ...data_asset.recipe import DatasetRecipe, RecipeManager
    rm = RecipeManager(R.dataset_dir(name))
    recipe = DatasetRecipe.from_dict(recipe_data)
    path = rm.save_recipe(recipe, recipe_name)
    return {"status": "saved", "path": str(path)}


# ── Recipe-aware Build ──

from pydantic import BaseModel
from typing import Optional

class RecipeBuildRequest(BaseModel):
    recipe_name: str = "default"
    output_dir: Optional[str] = None
    force: bool = False

@router.post("/datasets/{name}/build-recipe")
async def build_from_recipe(name: str, req: RecipeBuildRequest):
    """Build a training workspace from a Recipe.

    Uses the WorkspaceBuilder for full Recipe-driven build
    (multi-source, global dedup, camera-aware split).
    """
    from ...dataset_pipeline import DatasetPipeline
    from ...data_asset.profile import BuildProfile
    from pathlib import Path

    output_path = Path(req.output_dir) if req.output_dir else None
    profile = BuildProfile.detection_default(name=name)
    pipeline = DatasetPipeline()

    try:
        result = pipeline.run_build(
            dataset_name=name,
            recipe_name=req.recipe_name,
            profile=profile,
            output_dir=output_path,
            force=req.force,
        )
        return {
            "name": result.name,
            "output_dir": str(result.output_dir),
            "dataset_yaml": str(result.dataset_yaml),
            "total_images": result.total_images,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "test_count": result.test_count,
            "class_count": result.class_count,
            "class_names": result.class_names,
            "source_versions": result.source_versions,
            "skipped_blurry": result.skipped_blurry,
            "skipped_empty": result.skipped_empty,
            "skipped_duplicate": result.skipped_duplicate,
            "duration_sec": result.duration_sec,
            "created_at": result.created_at,
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(400, str(e))

# ── Tags (add/remove) ──

@router.delete("/datasets/{name}/tags/{tag_name}")
async def remove_tag(name: str, tag_name: str):
    """Remove a tag from a dataset."""
    tm = TagManager(R.dataset_dir(name))
    tm.undefine(tag_name)
    return {"status": "ok", "tag": tag_name}


# ── Misdata / Feedback (stats) ──

@router.get("/datasets/{name}/feedback/stats")
async def feedback_stats(name: str):
    """Get feedback statistics."""
    fm = FeedbackManager(name)
    items = fm.list_feedback()
    stats = {"false_positive": 0, "false_negative": 0, "hard_case": 0, "customer_case": 0}
    for item in items:
        cat = item.get("category", "false_positive")
        if cat in stats:
            stats[cat] += 1
    return stats


# ── Health Check Run ──

@router.post("/datasets/{name}/doctor")
async def run_doctor(name: str, data: dict = {}):
    """Run health check and return results."""
    version = data.get("version", "v1")
    doctor = DatasetDoctor(R.dataset_dir(name))
    import json as _json
    manifest_path = R.dataset_manifest_path(name, version)
    metadata_count = 0
    source_distribution = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata_count += 1
                    try:
                        rec = _json.loads(line)
                        src = rec.get("source", "unknown")
                        source_distribution[src] = source_distribution.get(src, 0) + 1
                    except _json.JSONDecodeError:
                        pass
    ann_dir = R.dataset_version_dir(name, version) / "annotations" / "converted" / "yolo"
    report = doctor.check(
        metadata_count=metadata_count,
        source_distribution=source_distribution,
        manifest_path=manifest_path,
        ann_dir=ann_dir if ann_dir.exists() else None,
    )
    return HealthReportSchema(
        dataset_name=name,
        version=version,
        created_at=report.created_at,
        overall_score=report.overall_score,
        checks=[{'name': c.name, 'status': c.status, 'score': c.score, 'message': c.message, 'detail': c.detail} for c in report.checks],
        issues=[c.message for c in report.checks if c.status != 'pass'],
        summary=report.summary,
        recommendations=report.recommendations,
    )

@router.get("/datasets/{name}/stats")
async def dataset_stats(name: str, version: str = Query("v1")):
    from ...data_asset.intelligence import DatasetIntelligence
    di = DatasetIntelligence(name, version)
    report = di.analyze()
    return AnnotationStatsSchema(
        total_images=report.total_images,
        annotated_images=report.annotated_images,
        background_images=report.background_images,
        total_annotations=report.total_annotations,
        avg_annotations_per_image=round(report.total_annotations / max(report.annotated_images, 1), 2),
        class_distribution=report.class_distribution,
        images_per_class=report.images_per_class,
    )


# ══════════════════════════════════════════════
#  Changelog
# ══════════════════════════════════════════════


@router.get("/datasets/{name}/changelog")
async def get_changelog(name: str, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Get dataset operation changelog (newest first)."""
    dataset_dir = R.dataset_dir(name)
    if not dataset_dir.exists():
        raise HTTPException(404, f"Dataset '{name}' not found")
    store = ChangelogStore(dataset_dir)
    entries = store.list(limit=limit, offset=offset)
    return {
        "dataset": name,
        "total": store.count(),
        "offset": offset,
        "limit": limit,
        "entries": [e.to_dict() for e in entries],
    }


# ══════════════════════════════════════════════
#  Dataset Browse & Annotation Editor Endpoints
# ══════════════════════════════════════════════

import json
from pathlib import Path
from PIL import Image as PILImage


def _load_manifest(version_dir: Path) -> list[dict]:
    """Load manifest.jsonl for a dataset version."""
    manifest_path = version_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_annotations(version_dir: Path, index: int) -> list[list[float]]:
    """Load YOLO annotations for an image by its manifest index."""
    yolo_dir = version_dir / "annotations" / "converted" / "yolo"
    ann_file = yolo_dir / f"{index:06d}.txt"
    if not ann_file.exists():
        return []
    boxes = []
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append([cls_id, xc, yc, w, h])
                except ValueError:
                    pass
    return boxes


def _save_annotations(version_dir: Path, index: int, boxes: list[list[float]]) -> None:
    """Save YOLO annotations for an image by its manifest index."""
    yolo_dir = version_dir / "annotations" / "converted" / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    ann_file = yolo_dir / f"{index:06d}.txt"
    with open(ann_file, "w") as f:
        for box in boxes:
            cls_id, xc, yc, w, h = box
            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image dimensions using PIL."""
    try:
        with PILImage.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


@router.get("/datasets/{name}/versions/{version}/browse")
async def browse_images(name: str, version: str):
    """List all images in a dataset version with annotation info."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    results = []
    for i, rec in enumerate(records):
        storage_path = rec.get("storage_path", "")
        img_path = R.resolve_storage_path(storage_path)
        boxes = _load_annotations(version_dir, i)
        width, height = 0, 0
        if img_path.exists():
            width, height = _get_image_dimensions(img_path)
        results.append({
            "index": i,
            "image_id": rec.get("image_id", f"img_{i:06d}"),
            "storage_path": storage_path,
            "width": width,
            "height": height,
            "annotation_count": len(boxes),
            "camera": rec.get("camera", ""),
            "capture_time": rec.get("capture_time", ""),
            "source": rec.get("source", ""),
        })
    return {
        "total": len(results),
        "annotated": sum(1 for r in results if r["annotation_count"] > 0),
        "images": results,
    }


@router.get("/datasets/{name}/versions/{version}/image/{index}")
async def serve_image(name: str, version: str, index: int):
    """Serve an image file by its manifest index."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    if index < 0 or index >= len(records):
        raise HTTPException(404, f"Image index {index} out of range (total: {len(records)})")

    storage_path = records[index].get("storage_path", "")
    img_path = R.resolve_storage_path(storage_path)
    if not img_path.exists():
        raise HTTPException(404, f"Image file not found: {storage_path}")

    return FileResponse(str(img_path), media_type="image/jpeg")


@router.get("/datasets/{name}/versions/{version}/annotations/{index}")
async def get_annotations(name: str, version: str, index: int):
    """Get YOLO annotations for a specific image."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    if index < 0 or index >= len(records):
        raise HTTPException(404, f"Image index {index} out of range")

    boxes = _load_annotations(version_dir, index)
    return {"index": index, "annotation_count": len(boxes), "boxes": boxes}


class UpdateAnnotationsRequest(BaseModel):
    boxes: list[list[float]]  # [[cls_id, xc, yc, w, h], ...]


@router.put("/datasets/{name}/versions/{version}/annotations/{index}")
async def update_annotations(name: str, version: str, index: int, req: UpdateAnnotationsRequest):
    """Save YOLO annotations for a specific image (overwrites all)."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    if index < 0 or index >= len(records):
        raise HTTPException(404, f"Image index {index} out of range")

    # Validate box format
    for box in req.boxes:
        if len(box) != 5:
            raise HTTPException(400, f"Each box must have 5 values [cls_id, xc, yc, w, h], got {len(box)}")
        cls_id, xc, yc, w, h = box
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            raise HTTPException(400, f"Box values must be in [0,1] range, got {box}")

    _save_annotations(version_dir, index, req.boxes)
    ChangelogStore(R.dataset_dir(name)).log_annotation_updated(version, index, len(req.boxes))
    return {"status": "saved", "index": index, "annotation_count": len(req.boxes)}


@router.get("/datasets/{name}/versions/{version}/classes")
async def get_classes(name: str, version: str):
    """Get unique class IDs present in all annotations of this version."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    yolo_dir = version_dir / "annotations" / "converted" / "yolo"
    all_class_ids: set[int] = set()
    if yolo_dir.exists():
        for ann_file in sorted(yolo_dir.glob("*.txt")):
            with open(ann_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            try:
                                all_class_ids.add(int(parts[0]))
                            except ValueError:
                                pass

    class_ids = sorted(all_class_ids)
    return {
        "total_classes": len(class_ids),
        "class_ids": class_ids,
        "classes": [{"id": cid, "name": f"class_{cid}"} for cid in class_ids],
    }


# ══════════════════════════════════════════════
#  AI Pre-labeling (P1-1)
#  Run a YOLO model to auto-generate annotations, then let humans
#  review/correct them — reduces manual effort by ~60%.
# ══════════════════════════════════════════════

class PreLabelRequest(BaseModel):
    model_path: str = Field(..., description="Path to a .pt YOLO model file")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
    imgsz: int = Field(640, ge=320, le=1920, description="Inference image size")
    indices: Optional[list[int]] = Field(
        None, description="Specific image indices to pre-label. None = all images."
    )
    overwrite: bool = Field(False, description="Overwrite existing annotations")


@router.post("/datasets/{name}/versions/{version}/prelabel")
async def prelabel_images(name: str, version: str, req: PreLabelRequest):
    """Run a YOLO model to auto-generate YOLO annotations for images.

    The generated annotations are saved as YOLO TXT files and can be
    subsequently reviewed/corrected via the annotation editor or review
    status endpoints.
    """
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(404, f"Model file not found: {req.model_path}")

    records = _load_manifest(version_dir)
    if not records:
        raise HTTPException(400, f"Dataset version has no images (empty manifest)")

    # Determine target indices
    target_indices = req.indices if req.indices is not None else list(range(len(records)))

    try:
        from ultralytics import YOLO
    except ImportError:
        raise HTTPException(500, "ultralytics is not installed in the environment")

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    labeled_count = 0
    skipped_count = 0
    total_boxes = 0

    for idx in target_indices:
        if idx < 0 or idx >= len(records):
            continue

        # Skip if annotations already exist and overwrite is False
        yolo_dir = version_dir / "annotations" / "converted" / "yolo"
        ann_file = yolo_dir / f"{idx:06d}.txt"
        if ann_file.exists() and not req.overwrite:
            skipped_count += 1
            continue

        # Resolve image path
        storage_path = records[idx].get("storage_path", "")
        img_path = R.resolve_storage_path(storage_path)
        if not img_path.exists():
            skipped_count += 1
            continue

        # Run inference
        try:
            results = model.predict(
                str(img_path),
                conf=req.conf,
                iou=req.iou,
                imgsz=req.imgsz,
                verbose=False,
            )
        except Exception:
            skipped_count += 1
            continue

        # Convert detections to YOLO format
        boxes_list: list[list[float]] = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                # Get image dimensions for normalization
                img_w, img_h = _get_image_dimensions(img_path)
                if img_w == 0 or img_h == 0:
                    img_w, img_h = 1920, 1080  # fallback

                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    xc = ((x1 + x2) / 2) / img_w
                    yc = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    boxes_list.append([cls_id, xc, yc, w, h])

        _save_annotations(version_dir, idx, boxes_list)
        labeled_count += 1
        total_boxes += len(boxes_list)

    ChangelogStore(R.dataset_dir(name)).append(ChangelogEntry(
        action="annotation.prelabel",
        detail=f"AI pre-label: {labeled_count} images, {total_boxes} boxes",
        version=version,
        count=total_boxes,
    ))

    return {
        "status": "completed",
        "labeled": labeled_count,
        "skipped": skipped_count,
        "total_boxes": total_boxes,
        "model_path": str(model_path),
        "conf": req.conf,
    }


# ══════════════════════════════════════════════
#  Per-image Review Status (P1-2)
#  Track annotation review progress: pending → approved/rejected
# ══════════════════════════════════════════════

_REVIEW_STATUSES = {"pending", "approved", "rejected"}


def _review_status_path(version_dir: Path) -> Path:
    """Path to the review status JSON file for a dataset version."""
    return version_dir / "review_status.json"


def _load_review_status(version_dir: Path) -> dict:
    """Load the review status map (index → status)."""
    path = _review_status_path(version_dir)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_review_status(version_dir: Path, status_map: dict) -> None:
    """Save the review status map."""
    path = _review_status_path(version_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status_map, f, ensure_ascii=False, indent=2)


@router.get("/datasets/{name}/versions/{version}/review-status")
async def get_review_status(name: str, version: str):
    """Get review status for all images in a dataset version."""
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    status_map = _load_review_status(version_dir)

    # Build per-image status list (default to "pending" for unreviewed)
    images = []
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for i in range(len(records)):
        status = status_map.get(str(i), "pending")
        if status not in _REVIEW_STATUSES:
            status = "pending"
        counts[status] = counts.get(status, 0) + 1
        images.append({"index": i, "status": status})

    return {
        "total": len(records),
        "summary": counts,
        "images": images,
    }


class ReviewStatusUpdate(BaseModel):
    status: str = Field(..., description="One of: pending, approved, rejected")
    reviewer: str = Field("", description="Optional reviewer name")


@router.put("/datasets/{name}/versions/{version}/review-status/{index}")
async def set_review_status(name: str, version: str, index: int, req: ReviewStatusUpdate):
    """Set the review status for a specific image."""
    if req.status not in _REVIEW_STATUSES:
        raise HTTPException(400, f"Invalid status '{req.status}'. Must be one of: {_REVIEW_STATUSES}")

    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    if index < 0 or index >= len(records):
        raise HTTPException(404, f"Image index {index} out of range (total: {len(records)})")

    status_map = _load_review_status(version_dir)
    status_map[str(index)] = req.status
    _save_review_status(version_dir, status_map)

    ChangelogStore(R.dataset_dir(name)).append(ChangelogEntry(
        action="annotation.review",
        detail=f"Image {index:06d} review status: {req.status}",
        version=version,
        user=req.reviewer,
    ))

    # Recompute summary
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for i in range(len(records)):
        s = status_map.get(str(i), "pending")
        if s in counts:
            counts[s] += 1

    return {
        "status": "updated",
        "index": index,
        "review_status": req.status,
        "summary": counts,
    }


class BatchReviewStatusUpdate(BaseModel):
    indices: list[int] = Field(..., description="Image indices to update")
    status: str = Field(..., description="One of: pending, approved, rejected")


@router.put("/datasets/{name}/versions/{version}/review-status")
async def batch_set_review_status(name: str, version: str, req: BatchReviewStatusUpdate):
    """Batch-set review status for multiple images at once."""
    if req.status not in _REVIEW_STATUSES:
        raise HTTPException(400, f"Invalid status '{req.status}'. Must be one of: {_REVIEW_STATUSES}")

    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    status_map = _load_review_status(version_dir)

    updated = 0
    for idx in req.indices:
        if 0 <= idx < len(records):
            status_map[str(idx)] = req.status
            updated += 1

    _save_review_status(version_dir, status_map)

    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for i in range(len(records)):
        s = status_map.get(str(i), "pending")
        if s in counts:
            counts[s] += 1

    return {
        "status": "updated",
        "updated_count": updated,
        "review_status": req.status,
        "summary": counts,
    }


# ══════════════════════════════════════════════
#  Batch Annotation Operations (P1-3)
#  Delete by class/area, reassign class, clear annotations
# ══════════════════════════════════════════════

class BatchAnnotationRequest(BaseModel):
    operation: str = Field(
        ...,
        description="One of: delete_by_class, delete_by_area, reassign_class, clear_images",
    )
    indices: Optional[list[int]] = Field(
        None, description="Image indices to process. None = all images."
    )
    # Parameters for delete_by_class
    class_id: Optional[int] = Field(None, description="Class ID for delete_by_class / reassign_class (source)")
    # Parameters for delete_by_area
    min_area: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Delete boxes with normalized area (w*h) below this threshold",
    )
    max_area: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Delete boxes with normalized area (w*h) above this threshold",
    )
    # Parameters for reassign_class
    new_class_id: Optional[int] = Field(
        None, description="Target class ID for reassign_class operation"
    )


@router.post("/datasets/{name}/versions/{version}/batch-annotations")
async def batch_annotation_operation(name: str, version: str, req: BatchAnnotationRequest):
    """Perform a batch annotation operation across multiple images.

    Supported operations:
    - **delete_by_class**: Remove all boxes with the given *class_id*.
    - **delete_by_area**: Remove boxes whose normalized area (w*h) falls
      outside [min_area, max_area].
    - **reassign_class**: Change all boxes of *class_id* to *new_class_id*.
    - **clear_images**: Remove all annotations from the specified images.
    """
    version_dir = R.dataset_version_dir(name, version)
    if not version_dir.exists():
        raise HTTPException(404, f"Version '{version}' of dataset '{name}' not found")

    records = _load_manifest(version_dir)
    if not records:
        raise HTTPException(400, "Dataset version has no images")

    target_indices = req.indices if req.indices is not None else list(range(len(records)))

    # Validate operation-specific parameters
    if req.operation in ("delete_by_class", "reassign_class") and req.class_id is None:
        raise HTTPException(400, f"class_id is required for operation '{req.operation}'")
    if req.operation == "reassign_class" and req.new_class_id is None:
        raise HTTPException(400, "new_class_id is required for reassign_class")
    if req.operation == "delete_by_area" and req.min_area is None and req.max_area is None:
        raise HTTPException(400, "At least one of min_area or max_area is required for delete_by_area")

    processed = 0
    boxes_removed = 0
    boxes_modified = 0
    images_cleared = 0

    for idx in target_indices:
        if idx < 0 or idx >= len(records):
            continue

        boxes = _load_annotations(version_dir, idx)
        if not boxes:
            continue

        original_count = len(boxes)
        new_boxes: list[list[float]] = []

        if req.operation == "delete_by_class":
            new_boxes = [b for b in boxes if int(b[0]) != req.class_id]
            boxes_removed += original_count - len(new_boxes)

        elif req.operation == "delete_by_area":
            for b in boxes:
                area = b[3] * b[4]  # w * h (normalized)
                if req.min_area is not None and area < req.min_area:
                    boxes_removed += 1
                    continue
                if req.max_area is not None and area > req.max_area:
                    boxes_removed += 1
                    continue
                new_boxes.append(b)

        elif req.operation == "reassign_class":
            new_boxes = []
            for b in boxes:
                if int(b[0]) == req.class_id:
                    new_boxes.append([req.new_class_id] + b[1:])
                    boxes_modified += 1
                else:
                    new_boxes.append(b)

        elif req.operation == "clear_images":
            new_boxes = []
            boxes_removed += original_count
            images_cleared += 1

        else:
            raise HTTPException(400, f"Unknown operation: {req.operation}")

        _save_annotations(version_dir, idx, new_boxes)
        processed += 1

    ChangelogStore(R.dataset_dir(name)).append(ChangelogEntry(
        action="annotation.batch",
        detail=f"batch:{req.operation} | processed={processed} removed={boxes_removed} modified={boxes_modified}",
        version=version,
        count=boxes_removed + boxes_modified,
    ))

    return {
        "status": "completed",
        "operation": req.operation,
        "images_processed": processed,
        "boxes_removed": boxes_removed,
        "boxes_modified": boxes_modified,
        "images_cleared": images_cleared,
    }

