"""Pipeline API — 数据生命周期工作流编排器。

Pipeline 是继 IngestionProfile / DatasetRecipe 之后的第三条对称管线：

  IngestionProfile        DatasetRecipe           PipelineProfile
       │                       │                       │
       ▼                       ▼                       ▼
  Raw Data ──→ Asset Dataset ──→ Training Workspace ──→ Trained Model
  storage/     datasets/         workspace/             projects/

PipelineProfile 是可保存、可复现的配置（data/pipeline/profiles/<name>.yaml），
与 IngestionProfile 和 DatasetRecipe 使用相同的管理模式。
"""

from __future__ import annotations

import json
import threading
import traceback as _tb
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...config import R, validate_name
from ...asset_manager import AssetManager
from ...data_asset.job import JobManager, JobStatus, JobType
from ...data_asset.version import VersionManager

router = APIRouter(tags=["pipeline"])


# ═══════════════════════════════════════════════════════════════
#  PipelineProfile CRUD
# ═══════════════════════════════════════════════════════════════

class StageConfig(BaseModel):
    enabled: bool = False
    raw_video_path: str = ""
    pipeline_stages: str = "motion,scene_change,burst,balance"
    profile_ref: str = ""


class StageDataset(BaseModel):
    name: str = ""
    task: str = "detection"
    source_dirs: List[str] = []


class StageAnnotation(BaseModel):
    auto_annotate: bool = False
    model: str = "yolo26n.pt"
    conf_threshold: float = 0.25


class StageBuild(BaseModel):
    enabled: bool = True
    recipe_ref: str = "default"
    output_dir: str = ""
    force: bool = False


class StageTraining(BaseModel):
    enabled: bool = False
    model: str = "yolo26n"
    epochs: int = 100
    batch: int = 16


class PipelineProfileRequest(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0"
    ingestion: StageConfig = StageConfig()
    dataset: StageDataset = StageDataset()
    annotation: StageAnnotation = StageAnnotation()
    build: StageBuild = StageBuild()
    training: StageTraining = StageTraining()


@router.get("/pipeline/profiles")
def list_pipeline_profiles():
    """列出所有 PipelineProfile。"""
    from ...data_asset.pipeline_profile import PipelineProfileManager
    mgr = PipelineProfileManager()
    names = mgr.list_profiles()
    jm = JobManager()
    # Map: profile_name → last job
    profile_jobs: dict = {}
    for job in jm.list_jobs(limit=200):
        if job.type != JobType.PIPELINE:
            continue
        pn = job.config.get("profile_name", "")
        if not pn:
            continue
        existing = profile_jobs.get(pn)
        if existing is None or job.created_at > existing["created_at"]:
            profile_jobs[pn] = {
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at,
            }
    profiles = []
    for name in names:
        p = mgr.get_profile(name)
        if p:
            d = p.to_dict()
            profiles.append({
                "name": d["name"],
                "description": d["description"],
                "dataset_name": d.get("dataset", {}).get("name", ""),
                "ingestion_enabled": d.get("ingestion", {}).get("enabled", False),
                "build_enabled": d.get("build", {}).get("enabled", True),
                "training_enabled": d.get("training", {}).get("enabled", False),
                "created_at": d["created_at"],
                "last_run": profile_jobs.get(name),
            })
    return {"profiles": profiles}


@router.get("/pipeline/profiles/{name}")
def get_pipeline_profile(name: str):
    """获取 PipelineProfile 详情。"""
    from ...data_asset.pipeline_profile import PipelineProfileManager
    mgr = PipelineProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        raise HTTPException(404, f"PipelineProfile '{name}' not found")
    return p.to_dict()


@router.post("/pipeline/profiles")
def create_pipeline_profile(req: PipelineProfileRequest):
    """创建 PipelineProfile。"""
    from ...data_asset.pipeline_profile import (
        PipelineProfile, PipelineProfileManager,
        PipelineIngestionConfig, PipelineDatasetConfig,
        PipelineAnnotationConfig, PipelineBuildConfig, PipelineTrainingConfig,
    )
    mgr = PipelineProfileManager()
    existing = mgr.get_profile(req.name)
    if existing:
        raise HTTPException(409, f"PipelineProfile '{req.name}' already exists")

    profile = PipelineProfile(
        name=req.name,
        version=req.version,
        description=req.description,
        ingestion=PipelineIngestionConfig(
            enabled=req.ingestion.enabled,
            raw_video_path=req.ingestion.raw_video_path,
            pipeline_stages=req.ingestion.pipeline_stages,
            profile_ref=req.ingestion.profile_ref,
        ),
        dataset=PipelineDatasetConfig(
            name=req.dataset.name or req.name,
            task=req.dataset.task,
            source_dirs=list(req.dataset.source_dirs),
        ),
        annotation=PipelineAnnotationConfig(
            auto_annotate=req.annotation.auto_annotate,
            model=req.annotation.model,
            conf_threshold=req.annotation.conf_threshold,
        ),
        build=PipelineBuildConfig(
            enabled=req.build.enabled,
            recipe_ref=req.build.recipe_ref,
            output_dir=req.build.output_dir,
            force=req.build.force,
        ),
        training=PipelineTrainingConfig(
            enabled=req.training.enabled,
            model=req.training.model,
            epochs=req.training.epochs,
            batch=req.training.batch,
        ),
    )
    path = mgr.save_profile(profile)
    return {"status": "created", "path": str(path), "profile": profile.to_dict()}


@router.put("/pipeline/profiles/{name}")
def update_pipeline_profile(name: str, req: PipelineProfileRequest):
    """更新 PipelineProfile。"""
    from ...data_asset.pipeline_profile import (
        PipelineProfile, PipelineProfileManager,
        PipelineIngestionConfig, PipelineDatasetConfig,
        PipelineAnnotationConfig, PipelineBuildConfig, PipelineTrainingConfig,
    )
    mgr = PipelineProfileManager()
    existing = mgr.get_profile(name)
    if existing is None:
        raise HTTPException(404, f"PipelineProfile '{name}' not found")

    profile = PipelineProfile(
        name=name,
        version=req.version,
        description=req.description,
        created_at=existing.created_at,
        ingestion=PipelineIngestionConfig(
            enabled=req.ingestion.enabled,
            raw_video_path=req.ingestion.raw_video_path,
            pipeline_stages=req.ingestion.pipeline_stages,
            profile_ref=req.ingestion.profile_ref,
        ),
        dataset=PipelineDatasetConfig(
            name=req.dataset.name or name,
            task=req.dataset.task,
            source_dirs=list(req.dataset.source_dirs),
        ),
        annotation=PipelineAnnotationConfig(
            auto_annotate=req.annotation.auto_annotate,
            model=req.annotation.model,
            conf_threshold=req.annotation.conf_threshold,
        ),
        build=PipelineBuildConfig(
            enabled=req.build.enabled,
            recipe_ref=req.build.recipe_ref,
            output_dir=req.build.output_dir,
            force=req.build.force,
        ),
        training=PipelineTrainingConfig(
            enabled=req.training.enabled,
            model=req.training.model,
            epochs=req.training.epochs,
            batch=req.training.batch,
        ),
    )
    path = mgr.save_profile(profile)
    return {"status": "updated", "path": str(path), "profile": profile.to_dict()}


@router.delete("/pipeline/profiles/{name}")
def delete_pipeline_profile(name: str):
    """删除 PipelineProfile。"""
    from ...data_asset.pipeline_profile import PipelineProfileManager
    mgr = PipelineProfileManager()
    p = mgr.get_profile(name)
    if p is None:
        raise HTTPException(404, f"PipelineProfile '{name}' not found")
    mgr.delete_profile(name)
    return {"status": "deleted", "name": name}


@router.post("/pipeline/profiles/{name}/run")
def run_pipeline_profile(name: str):
    """执行 PipelineProfile（异步 Job 模式）。"""
    from ...data_asset.pipeline_profile import PipelineProfileManager
    mgr = PipelineProfileManager()
    profile = mgr.get_profile(name)
    if profile is None:
        raise HTTPException(404, f"PipelineProfile '{name}' not found")

    params = mgr.profile_to_request(profile)
    job = _start_pipeline_job(params, profile_name=name)
    return job


# ═══════════════════════════════════════════════════════════════
#  Inline Pipeline Run (backward-compatible, no profile needed)
# ═══════════════════════════════════════════════════════════════

class FullPipelineRequest(BaseModel):
    """全生命周期管道请求参数（也可通过 PipelineProfile 管理）。"""
    dataset_name: str
    task: str = "detection"
    source_dirs: List[str] = []
    raw_video_path: Optional[str] = None
    pipeline_stages: str = "motion,scene_change,burst,balance"
    auto_annotate: bool = False
    model_name: str = "yolo26n.pt"
    conf_threshold: float = 0.25
    build_workspace: bool = True
    output_dir: Optional[str] = None
    force_build: bool = False
    start_training: bool = False
    training_model: str = "yolo26n"
    training_epochs: int = 100
    training_batch: int = 16


@router.post("/pipeline/run")
def run_full_pipeline(req: FullPipelineRequest):
    """运行完整数据生命周期管道（异步 Job 模式）。"""
    job = _start_pipeline_job(req.model_dump())
    return job


def _start_pipeline_job(params: dict, profile_name: str = "") -> dict:
    """Create and start a Pipeline Job with the given parameters."""
    # Validate the dataset name before starting a background thread: a bad name
    # used to fail several minutes into the run, inside the worker.
    dataset_name = params.get("dataset_name", "")
    if not dataset_name:
        raise HTTPException(400, "dataset_name is required")
    validate_name(dataset_name, "dataset name")

    jm = JobManager()
    config = {
        "dataset_name": params.get("dataset_name", ""),
        "task": params.get("task", "detection"),
        "source_dirs": params.get("source_dirs", []),
        "raw_video_path": params.get("raw_video_path"),
        "auto_annotate": params.get("auto_annotate", False),
        "build_workspace": params.get("build_workspace", True),
        "start_training": params.get("start_training", False),
    }
    if profile_name:
        config["profile_name"] = profile_name

    label = f"Pipeline: {profile_name or config['dataset_name']}"
    job = jm.create(job_type=JobType.PIPELINE, config=config, label=label)

    thread = threading.Thread(
        target=_pipeline_thread,
        args=(job.job_id, params),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "job_id": job.job_id,
        "message": f"Pipeline job {job.job_id} started. Track on jobs page.",
    }


def _pipeline_thread(job_id: str, params: dict):
    """后台执行完整管道。"""
    from ...data_asset import DatasetRegistry, AssetStore, DatasetDoctor, LineageStore, LineageNode

    jm = JobManager()

    def _log(msg: str):
        jm.append_log(job_id, msg)

    try:
        jm.update_status(job_id, JobStatus.RUNNING)
        dataset_name = params.get("dataset_name", "unnamed")
        task = params.get("task", "detection")
        source_dirs = params.get("source_dirs", [])
        raw_video_path = params.get("raw_video_path")
        pipeline_stages = params.get("pipeline_stages", "motion,scene_change,burst,balance")
        auto_annotate = params.get("auto_annotate", False)
        model_name = params.get("model_name", "yolo26n.pt")
        conf_threshold = params.get("conf_threshold", 0.25)
        build_workspace = params.get("build_workspace", True)
        force_build = params.get("force_build", False)
        start_training = params.get("start_training", False)

        _log(f"[Pipeline] Starting: {dataset_name}")

        # ── Step 0: Ingestion (optional, from raw video) ──
        if raw_video_path:
            from ...ingestion.pipeline import IngestionPipeline
            from ...ingestion.core import QualityFilterConfig, DedupConfig
            from ...ingestion.sampling.core import DecodeConfig

            _log(f"[Step 0] Ingestion from: {raw_video_path}")
            stages = [s.strip() for s in pipeline_stages.split(",")]
            quality = QualityFilterConfig(blur_laplacian_threshold=100.0)
            dedup = DedupConfig(method="dhash", hash_distance=5)
            decode = DecodeConfig(mode="interval", every_n_frames=30, jpeg_quality=80)

            ingest_pipeline = IngestionPipeline(
                raw_root=raw_video_path,
                sampling_pipeline=stages,
                quality=quality,
                dedup=dedup,
                decode_config=decode,
                auto_register=True,
                target_dataset=dataset_name,
            )
            report = ingest_pipeline.run(
                output_dir=R.ingestion_run_dir(f"pipeline_{job_id}"),
                skip_scene=True,
                generate_report=False,
            )
            _log(f"  Ingestion complete: {report.total_after_dedup} frames kept")

        # ── Step 1: Create dataset ──
        store = AssetStore(R.DATASETS)
        registry = DatasetRegistry(store)
        try:
            registry.create(name=dataset_name, task=task)
            _log(f"[Step 1] Dataset created: {dataset_name}")
        except FileExistsError:
            _log(f"[Step 1] Dataset already exists: {dataset_name}")

        # ── Step 2: Register Manifest ──
        # Versions are append-only and cumulative: create a new snapshot through
        # VersionManager instead of overwriting ``v1/manifest.jsonl`` (which both
        # bypassed the version metadata and could destroy an existing snapshot).
        am = AssetManager()
        project_root = R.PROJECT_ROOT.resolve()
        records = []
        storage = R.STORAGE_IMAGES
        extensions = {".jpg", ".jpeg", ".png"}
        target_dirs = [d for d in sorted(storage.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        if source_dirs:
            target_dirs = [d for d in target_dirs if d.name in source_dirs]
        for src_dir in target_dirs:
            for f in sorted(src_dir.rglob("*")):
                if f.suffix.lower() not in extensions:
                    continue
                try:
                    rel = f.resolve().relative_to(project_root)
                    records.append(am.create_manifest_record(
                        storage_path=str(rel.as_posix()), source=src_dir.name))
                except ValueError:
                    continue

        version = ""
        if records:
            vm = VersionManager(R.dataset_dir(dataset_name))
            snapshot = vm.create_version(
                manifest_records=records,
                description=f"Pipeline {job_id}",
                source="pipeline",
            )
            version = snapshot.version
            _log(f"[Step 2] Manifest registered: {len(records)} images -> {version}")
        else:
            _log("[Step 2] No source images found; manifest not created")

        version_dir = R.dataset_version_dir(dataset_name, version) if version else None

        # ── Step 3: Auto-annotate (optional) ──
        if auto_annotate and records and version_dir is not None:
            from ultralytics import YOLO
            import cv2
            _log(f"[Step 3] Auto-annotating with {model_name}...")
            model = YOLO(model_name)
            ann_dir = version_dir / "annotations" / "converted" / "yolo"
            ann_dir.mkdir(parents=True, exist_ok=True)
            annotated = 0
            for i, rec in enumerate(records):
                img_path = project_root / rec["storage_path"]
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                results = model(img, verbose=False, conf=conf_threshold)
                if results and results[0].boxes is not None:
                    with open(ann_dir / "{i:06d}.txt", "w", encoding="utf-8") as lf:
                        for box, cls_id in zip(results[0].boxes.xyxyn, results[0].boxes.cls):
                            x1, y1, x2, y2 = box.tolist()
                            lf.write("{int(cls_id)} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
                annotated += 1
            _log(f"  Annotated: {annotated} / {len(records)} images")

        # ── Step 4: Health Check ──
        if records and version_dir is not None:
            try:
                doctor = DatasetDoctor(R.dataset_dir(dataset_name))
                ann_dir = version_dir / "annotations" / "converted" / "yolo"
                health = doctor.check(
                    metadata_count=len(records),
                    manifest_path=version_dir / "manifest.jsonl",
                    ann_dir=ann_dir if ann_dir.exists() else None,
                )
                _log("[Step 4] Health score: {health.overall_score:.0f}/100")
            except Exception as e:
                _log(f"[Step 4] Health check skipped: {e}")

        # ── Step 5: Build Workspace ──
        if build_workspace and records:
            from ...dataset_pipeline import DatasetPipeline
            from ...data_asset.profile import BuildProfile
            _log("[Step 5] Building workspace from recipe...")
            profile = BuildProfile.detection_default(name=dataset_name)
            pipeline = DatasetPipeline()
            try:
                result = pipeline.run_build(
                    dataset_name=dataset_name,
                    recipe_name=params.get("recipe_ref", "default"),
                    profile=profile,
                    force=force_build,
                )
                _log(f"  Workspace: {result.train_count} train / {result.val_count} val / {result.test_count} test")
                _log(f"  dataset.yaml: {result.dataset_yaml}")
            except FileNotFoundError:
                _log("  [WARN] No recipe found. Create with 'dataset recipe --create' first.")
            except FileExistsError:
                _log("  [WARN] Workspace exists. Use force_build=True to overwrite.")

        # ── Step 6: Record Lineage ──
        try:
            lineage_store = LineageStore(R.dataset_dir(dataset_name))
            lineage_store.add_node(LineageNode(
                id=f"pipeline_{job_id}",
                type="pipeline",
                name=f"Pipeline run {job_id}",
                metadata={"source": "pipeline", "auto_annotate": auto_annotate, "build_workspace": build_workspace},
            ))
            _log("[Step 6] Lineage recorded")
        except Exception as e:
            _log(f"[Step 6] Lineage skipped: {e}")

        # ── Step 7: Start Training (optional) ──
        if start_training:
            from ...cli.new import ProjectScaffolder
            from ...cli.train import _do_train

            _log("[Step 7] Creating training project...")
            try:
                proj_name = f"{dataset_name}_model"
                scaffolder = ProjectScaffolder(
                    name=proj_name,
                    task=task,
                    backend="ultralytics",
                    model=params.get("training_model", "yolo26n"),
                    dataset_source="local",
                )
                project_dir = scaffolder.run()
                if build_workspace:
                    ws_dataset_yaml = R.workspace_dir(f"{dataset_name}_default") / "dataset.yaml"
                    if ws_dataset_yaml.exists():
                        import shutil
                        shutil.copy2(str(ws_dataset_yaml), str(project_dir / "dataset.yaml"))
                _log(f"  Project created: {proj_name}")
                _log("  Starting training...")
                result = _do_train(
                    config_path=str(project_dir / "project.yaml"),
                    device=params.get("training_device", "cuda:0"),
                    epochs=params.get("training_epochs", 100),
                    batch=params.get("training_batch", 16),
                    log_func=lambda msg: _log(f"  [Train] {msg}"),
                )
                _log(f"  Training complete. Model: {result.get('best_model', 'N/A')}")
            except FileExistsError:
                _log("  [WARN] Training project already exists. Skipping.")
            except Exception as e:
                _log(f"  [WARN] Training skipped: {e}")

        # ── Done ──
        _log("[Pipeline] Complete!")
        jm.save_result(job_id, {
            "dataset_name": dataset_name,
            "total_images": len(records) if records else 0,
            "auto_annotated": auto_annotate,
            "workspace_built": build_workspace,
            "training_started": start_training,
        })
        jm.update_status(job_id, JobStatus.COMPLETED)

    except Exception as e:
        error_msg = f"[Pipeline] FAILED: {e}\n" + _tb.format_exc()
        _log(error_msg)
        jm.save_result(job_id, {"error": str(e)})
        jm.update_status(job_id, JobStatus.FAILED, error=str(e))


# ═══════════════════════════════════════════════════════════════
#  Low-level utility endpoints
# ═══════════════════════════════════════════════════════════════


@router.get("/pipeline/storage-sources")
def list_storage_sources():
    """列出 storage/images/ 下的图片源目录。"""
    sources = []
    storage = R.STORAGE_IMAGES
    if storage.exists():
        for d in sorted(storage.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            img_count = len([f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")])
            if img_count == 0:
                img_count = len([f for f in d.rglob("*") if f.is_file()])
            sources.append({"name": d.name, "path": str(d.resolve()), "image_count": img_count})
    return sources


class RegisterManifestRequest(BaseModel):
    dataset_name: str
    version: str = "v1"
    source_dirs: List[str] = []


@router.post("/pipeline/register-manifest")
def register_manifest(req: RegisterManifestRequest):
    """扫描 storage/images/ 注册 Manifest 到数据集版本。"""
    am = AssetManager()
    project_root = R.PROJECT_ROOT.resolve()
    records = []
    storage = R.STORAGE_IMAGES
    if not storage.exists():
        raise HTTPException(400, "Storage directory not found: " + str(storage))
    target_dirs = [d for d in sorted(storage.iterdir()) if d.is_dir()]
    if req.source_dirs:
        target_dirs = [d for d in target_dirs if d.name in req.source_dirs]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for src_dir in target_dirs:
        if src_dir.name.startswith("."):
            continue
        for f in sorted(src_dir.rglob("*")):
            if f.suffix.lower() not in extensions:
                continue
            try:
                rel = f.resolve().relative_to(project_root)
                records.append(am.create_manifest_record(storage_path=str(rel.as_posix()), source=src_dir.name))
            except ValueError:
                continue
    if not records:
        raise HTTPException(400, "No images found in the specified source directories")
    manifest_path = R.dataset_manifest_path(req.dataset_name, req.version)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    am.save_manifest(records, manifest_path)
    return {"status": "ok", "dataset": req.dataset_name, "version": req.version,
            "total_images": len(records), "manifest_path": str(manifest_path)}


class AutoAnnotateRequest(BaseModel):
    dataset_name: str
    version: str = "v1"
    model_name: str = "yolo26n.pt"
    conf_threshold: float = 0.25
    max_images: int = 0


@router.post("/pipeline/auto-annotate")
def auto_annotate(req: AutoAnnotateRequest):
    """用 YOLO 模型自动标注未标记图片。"""
    manifest_path = R.dataset_manifest_path(req.dataset_name, req.version)
    if not manifest_path.exists():
        raise HTTPException(400, f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = [json.loads(l) for l in f if l.strip()]
    if req.max_images > 0:
        manifest = manifest[:req.max_images]
    ann_dir = R.dataset_version_dir(req.dataset_name, req.version) / "annotations" / "converted" / "yolo"
    ann_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    import cv2
    model = YOLO(req.model_name)
    project_root = R.PROJECT_ROOT.resolve()
    annotated = 0
    errors = 0
    for i, rec in enumerate(manifest):
        img_path = project_root / rec["storage_path"]
        if not img_path.exists():
            errors += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue
        results = model(img, verbose=False, conf=req.conf_threshold)
        label_path = ann_dir / "{i:06d}.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            for r in results:
                if r.boxes is None:
                    continue
                for box, cls_id in zip(r.boxes.xyxyn, r.boxes.cls):
                    x1, y1, x2, y2 = box.tolist()
                    lf.write("{int(cls_id)} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
        annotated += 1
    return {"status": "ok", "dataset": req.dataset_name, "version": req.version,
            "total_images": len(manifest), "annotated": annotated, "errors": errors,
            "annotation_dir": str(ann_dir)}


class HealthCheckRequest(BaseModel):
    dataset_name: str
    version: str = "v1"


@router.post("/pipeline/health-check")
def health_check(req: HealthCheckRequest):
    """对数据集版本执行健康检查。"""
    from ...data_asset import DatasetDoctor
    asset_dir = R.dataset_dir(req.dataset_name)
    if not asset_dir.exists():
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found")
    doctor = DatasetDoctor(asset_dir)
    manifest_path = R.dataset_manifest_path(req.dataset_name, req.version)
    metadata_count = 0
    source_distribution: Dict[str, int] = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata_count += 1
                    try:
                        rec = json.loads(line)
                        src = rec.get("source", "unknown")
                        source_distribution[src] = source_distribution.get(src, 0) + 1
                    except json.JSONDecodeError:
                        pass
    ann_dir = R.dataset_version_dir(req.dataset_name, req.version) / "annotations" / "converted" / "yolo"
    report = doctor.check(metadata_count=metadata_count, source_distribution=source_distribution,
                          manifest_path=manifest_path, ann_dir=ann_dir if ann_dir.exists() else None)
    return {"status": "ok", "dataset": req.dataset_name, "version": req.version,
            "overall_score": report.overall_score, "summary": report.summary,
            "checks": [{"name": c.name, "status": c.status, "score": c.score, "message": c.message} for c in report.checks],
            "recommendations": report.recommendations}


# ═══════════════════════════════════════════════════════════════
#  Pipeline run status (delegates to Job system)
# ═══════════════════════════════════════════════════════════════

@router.get("/pipeline/runs")
def list_pipeline_runs(limit: int = Query(50)):
    """列出所有 Pipeline Job 运行记录。"""
    jm = JobManager()
    jobs = [j for j in jm.list_jobs(limit=limit) if j.type == JobType.PIPELINE]
    return [{
        "run_id": j.job_id,
        "status": j.status.value,
        "dataset": j.config.get("dataset_name", ""),
        "profile": j.config.get("profile_name", ""),
        "created_at": j.created_at[:19],
        "log_count": _count_log_lines(j.log_path) if j.log_path and Path(j.log_path).exists() else 0,
    } for j in jobs]


def _count_log_lines(log_path: str) -> int:
    """Count lines in a log file without leaking the file handle (M-10)."""
    with open(log_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


@router.get("/pipeline/runs/{run_id}")
def get_pipeline_run(run_id: str):
    """获取 Pipeline 运行详情。"""
    jm = JobManager()
    job = jm.get(run_id)
    if job is None:
        raise HTTPException(404, f"Pipeline run '{run_id}' not found")
    log_content = ""
    if job.log_path and Path(job.log_path).exists():
        log_content = Path(job.log_path).read_text(encoding="utf-8")
    result = {}
    if job.result_path and Path(job.result_path).exists():
        result = json.loads(Path(job.result_path).read_text(encoding="utf-8"))
    return {
        "run_id": job.job_id,
        "status": job.status.value,
        "config": job.config,
        "created_at": job.created_at[:19] if job.created_at else "",
        "finished_at": job.finished_at[:19] if job.finished_at else "",
        "error": job.error,
        "log": log_content,
        "result": result,
    }
