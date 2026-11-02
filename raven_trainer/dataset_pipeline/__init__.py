"""Dataset Pipeline — 数据集流水线编排层。

这是 Recipe → Workspace 的高层编排器，负责：
  1. 创建 Job 跟踪构建过程
  2. 调用 WorkspaceBuilder 执行构建
  3. 保存构建结果到 lineage
  4. 支持 CLI 和 API 调用

与 data_asset/workspace/ 的区别：
  workspace/          = 低层引擎（Builder）
  dataset_pipeline/   = 高层编排（Orchestrator + Job + Lineage）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import R
from ..data_asset.job import JobManager, Job, JobStatus, JobType
from ..data_asset.recipe import DatasetRecipe, RecipeSource, RecipeSplit, RecipeManager
from ..data_asset.profile import BuildProfile
from ..data_asset.workspace import WorkspaceBuilder, WorkspaceBuildResult
from ..data_asset.lineage import LineageStore, LineageNode, LineageEdge


@dataclass
class PipelineBuildConfig:
    """Configuration for a single pipeline build run."""
    dataset_name: str
    recipe_name: str = "default"
    profile_name: str = "default"
    output_dir: Optional[str] = None
    force: bool = False
    seed: int = 42
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "recipe_name": self.recipe_name,
            "profile_name": self.profile_name,
            "output_dir": self.output_dir,
            "force": self.force,
            "seed": self.seed,
            "description": self.description,
        }


class DatasetPipeline:
    """数据集流水线——高层编排器。

    这个类是 dataset recipe --build 命令和未来 Web UI 的入口。

    Usage:
        pipeline = DatasetPipeline()
        result = pipeline.run_build(
            dataset_name="person_detection",
            recipe_name="default",
        )
    """

    def __init__(self):
        self.job_manager = JobManager()

    def run_build(
        self,
        dataset_name: str,
        recipe_name: str = "default",
        profile: Optional[BuildProfile] = None,
        output_dir: Optional[Path] = None,
        force: bool = False,
        seed: int = 42,
        description: str = "",
    ) -> WorkspaceBuildResult:
        """Run a full build: log job → build workspace → save result.

        Returns:
            WorkspaceBuildResult from the builder
        """
        # 1. Create job
        config_dict = {
            "dataset_name": dataset_name,
            "recipe_name": recipe_name,
            "output_dir": str(output_dir) if output_dir else None,
            "force": force,
            "seed": seed,
        }
        job = self.job_manager.create(
            job_type=JobType.BUILD,
            config=config_dict,
            label=description or f"Build {dataset_name}@{recipe_name}",
        )

        try:
            self.job_manager.update_status(job.job_id, JobStatus.RUNNING)
            self.job_manager.append_log(job.job_id,
                                        f"Starting build: {dataset_name} recipe={recipe_name}")

            # 2. Run the builder
            builder = WorkspaceBuilder(dataset_name=dataset_name)
            result = builder.build(
                recipe_name=recipe_name,
                profile=profile,
                output_dir=output_dir,
                force=force,
                seed=seed,
            )

            # 3. Save result
            result_dict = {
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
            self.job_manager.save_result(job.job_id, result_dict)
            self.job_manager.append_log(
                job.job_id,
                f"Build complete: {result.train_count} train, {result.val_count} val, "
                f"{result.test_count} test, {result.class_count} classes "
                f"({result.duration_sec}s)",
            )
            self.job_manager.update_status(job.job_id, JobStatus.COMPLETED)

            return result

        except Exception as e:
            self.job_manager.append_log(job.job_id, f"Build failed: {e}")
            self.job_manager.update_status(job.job_id, JobStatus.FAILED, error=str(e))
            raise

    def list_builds(self, dataset_name: Optional[str] = None, limit: int = 20) -> List[Job]:
        """List all build jobs, optionally filtered by dataset."""
        jobs = self.job_manager.list_jobs(status=None, limit=limit)
        if dataset_name:
            jobs = [j for j in jobs
                    if j.type == JobType.BUILD
                    and j.config.get("dataset_name") == dataset_name]
        else:
            jobs = [j for j in jobs if j.type == JobType.BUILD]
        return jobs


# ── Singleton convenience ──

_default_pipeline = None


def get_pipeline() -> DatasetPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DatasetPipeline()
    return _default_pipeline


def run_build(
    dataset_name: str,
    recipe_name: str = "default",
    profile: Optional[BuildProfile] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    seed: int = 42,
) -> WorkspaceBuildResult:
    """Convenience: get pipeline and run build."""
    return get_pipeline().run_build(
        dataset_name=dataset_name,
        recipe_name=recipe_name,
        profile=profile,
        output_dir=output_dir,
        force=force,
        seed=seed,
    )
