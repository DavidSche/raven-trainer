"""Data Ingestion Pipeline.

Raw Data
    ↓
Import (建立索引, metadata.json)
    ↓
Frame Sampling (自适应抽帧 / 按事件抽帧)
    ↓
Quality Check (模糊/曝光/黑屏/遮挡/分辨率)
    ↓
Deduplicate (dHash / pHash / SSIM 感知去重)
    ↓
Scene Analysis (聚类 + 分布统计)
    ↓
Dataset Report (自动生成 HTML 数据质量报告)
    ↓
Annotation (CVAT / Label Studio)
"""

from .profile import IngestionProfile, IngestionProfileManager, SourceRef
from .core import (
    RawSource, RawSourceMetadata,
    ImportedSample,
    IngestionReport,
    SamplingStrategy,
    QualityFilterConfig,
    DedupConfig,
)
from .pipeline import IngestionPipeline
from .sampling.core import DecodeConfig

__all__ = [
    "IngestionProfile", "IngestionProfileManager", "SourceRef",
    "RawSource", "RawSourceMetadata",
    "ImportedSample",
    "IngestionPipeline",
    "IngestionReport",
    "SamplingStrategy",
    "QualityFilterConfig",
    "DedupConfig",
    "DecodeConfig",
]
