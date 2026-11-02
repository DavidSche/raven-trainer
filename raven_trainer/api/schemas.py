"""Pydantic schemas for REST API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Dataset ──

class DatasetSchema(BaseModel):
    name: str
    display_name: str
    task: str
    status: str
    current_version: str
    version_count: int
    total_images: int
    total_annotations: int
    health_score: float
    created_at: str
    updated_at: str
    tags: List[str] = []
    owner: str = ""

class VersionSchema(BaseModel):
    version: str
    image_count: int
    annotation_count: int
    source: str
    created_at: str
    parent_version: Optional[str] = None

class TagSchema(BaseModel):
    name: str
    color: str = ""
    description: str = ""

class AnnotationStatsSchema(BaseModel):
    total_images: int
    annotated_images: int
    background_images: int
    total_annotations: int
    avg_annotations_per_image: float
    class_distribution: Dict[str, int] = {}
    images_per_class: Dict[str, int] = {}

class AnnotationTaskSchema(BaseModel):
    task_id: str
    platform: str
    name: str
    status: str
    image_count: int
    label_count: int
    created_at: str
    completed_at: Optional[str] = None
    notes: str = ""

# ── Intelligence Report ──

class IntelligenceReportSchema(BaseModel):
    dataset_name: str
    version: str
    created_at: str
    total_images: int
    total_annotations: int
    annotated_images: int
    background_images: int
    background_ratio: float
    class_distribution: Dict[str, int] = {}
    camera_distribution: Dict[str, int] = {}
    camera_bias_score: float = 100.0
    time_distribution: Dict[str, int] = {}
    object_size_bins: Dict[str, int] = {}
    small_object_ratio: float = 0.0
    medium_object_ratio: float = 0.0
    large_object_ratio: float = 0.0
    bias_warnings: List[str] = []
    recommendations: List[str] = []

# ── Health Report ──

class HealthReportSchema(BaseModel):
    dataset_name: str
    version: str = ""
    created_at: str = ""
    overall_score: float = 0.0
    checks: List[Dict[str, Any]] = []
    issues: List[str] = []
    summary: str = ""
    recommendations: List[str] = []

# ── Job ──

class JobSchema(BaseModel):
    job_id: str
    type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    config: Dict[str, Any] = {}

class JobLogSchema(BaseModel):
    job_id: str
    log: str

# ── Training / Experiment ──

class ExperimentSchema(BaseModel):
    id: str
    project: str
    model: str
    backend: str
    status: str
    epochs: int
    batch: int
    imgsz: int
    device: str = "cuda:0"
    metrics: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    created_at: str
    finished_at: Optional[str] = None
    dataset: str = ""
    workspace: str = ""

class ExperimentConfigSchema(BaseModel):
    project: str
    model: str = "yolo26n"
    backend: str = "ultralytics"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    dataset: str = ""
    workspace: str = ""
    device: str = "cuda:0"
    export_format: str = "onnx"
    export_opset: int = 17
    export_half: bool = False
    export_int8: bool = False
    export_dynamic: bool = False
    export_simplify: bool = True
    register_to_platform: bool = False
    # Per-backend training hyperparameters from the New Training form.
    # Merged into train_cfg by _do_train; each adapter's kwarg whitelist is the
    # final guard (unknown keys are dropped, not forwarded to the backend).
    hyperparameters: Dict[str, Any] = {}

# ── Model ──

class ModelSchema(BaseModel):
    id: str
    project: str
    version: str
    type: str
    format: str
    path: str
    metrics: Dict[str, Any] = {}
    created_at: str
    experiment_id: str = ""

# ── Feedback / Misdata ──

class MisdataItemSchema(BaseModel):
    filename: str
    category: str
    status: str
    path: str

# ── Ingestion ──

class IngestionRunSchema(BaseModel):
    run_id: str
    status: str
    source_count: int
    total_frames: int
    kept_frames: int
    created_at: str
    pipeline: str = ""
    report_path: str = ""

# ── Lineage ──

class LineageNodeSchema(BaseModel):
    id: str
    type: str  # dataset, version, experiment, model, workspace
    label: str
    created_at: str = ""

class LineageSchema(BaseModel):
    nodes: List[LineageNodeSchema] = []
    edges: List[Dict[str, str]] = []

# ── Storage ──

class StorageSourceSchema(BaseModel):
    name: str
    type: str  # video, image, directory
    path: str
    frames: int = 0
    resolution: str = ""
    duration: str = ""
