"""Data Asset Management — 工业视觉数据资产管理平台。

九大模块：
  Registry    — 数据集注册中心（类似 Git Repository）
  Version     — 不可变版本管理（Append-Only）
  Metadata    — 每张图片的结构化元数据
  Tag         — 标签系统（Tag ≠ Label）
  Quality     — 质量数据库（预留）
  Statistics  — 自动实时统计（通过 Metadata 实现）
  Recipe      — 数据集构建规则（数据集的"源码"）
  Workspace   — 根据 Recipe 构建训练工作区（已完成）
  Lineage     — 数据血缘追踪（Video → Dataset → ONNX）
  Doctor      — 数据集健康检查（Health Score 0-100）
"""

from .registry import DatasetRegistry, DataAsset, AssetStatus, AssetStore
from .version import VersionManager, VersionSnapshot
from .metadata import MetadataStore, ImageMetadata
from .tag import TagManager, TagDef
from .recipe import DatasetRecipe, RecipeSource, RecipeSplit, RecipeAugment, RecipeManager
from .lineage import LineageStore, LineageNode, LineageEdge
from .annotation import AnnotationManager, AnnotationTask, TaskManager
from .intelligence import DatasetIntelligence, IntelligenceReport
from .feedback import FeedbackManager
from .job import JobManager, Job, JobStatus, JobType
from .profile import BuildProfile
from .changelog import ChangelogStore, ChangelogEntry
from .doctor import DatasetDoctor, HealthReport, HealthCheckItem
from .split import DatasetSplitter
from .misdata import MisdataManager
from .workspace import WorkspaceBuilder, WorkspaceBuildResult, WorkspaceSample, build_workspace

__all__ = [
    "DatasetSplitter", "MisdataManager",
    "WorkspaceBuilder", "WorkspaceBuildResult", "WorkspaceSample", "build_workspace",
    "DatasetRegistry", "DataAsset", "AssetStatus", "AssetStore",
    "VersionManager", "VersionSnapshot",
    "MetadataStore", "ImageMetadata",
    "TagManager", "TagDef",
    "DatasetRecipe", "RecipeSource", "RecipeSplit", "RecipeAugment", "RecipeManager",
    "LineageStore", "LineageNode", "LineageEdge",
    "AnnotationManager", "AnnotationTask", "TaskManager",
    "DatasetIntelligence", "IntelligenceReport",
    "FeedbackManager",
    "JobManager", "Job", "JobStatus", "JobType",
    "BuildProfile",
    "DatasetDoctor", "HealthReport", "HealthCheckItem",
    "ChangelogStore", "ChangelogEntry",
]
