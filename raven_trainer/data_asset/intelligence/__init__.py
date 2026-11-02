"""Dataset Intelligence — 数据智能分析层。

分析数据集的分布、质量、偏置，不依赖任何训练框架。
所有分析基于 Manifest、Metadata、Annotation，是 Dataset Doctor 的增强层。

能力：
  - Class Distribution（类别分布）
  - Camera Distribution（摄像头分布）
  - Time Distribution（昼夜/班次）
  - Occlusion Distribution（遮挡等级）
  - Object Size Distribution（目标尺寸）
  - Empty / Background Ratio（背景图比例）
  - Hard Example Mining（困难样本发现）
  - Dataset Bias Detection（数据偏置分析）

使用方式：
    from raven_trainer.data_asset.intelligence import DatasetIntelligence
    di = DatasetIntelligence("workshop", "v1")
    report = di.analyze()
    print(report.summary())
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config import R


@dataclass
class IntelligenceReport:
    """数据智能分析报告。"""

    dataset_name: str
    version: str
    created_at: str = ""

    # 基本统计
    total_images: int = 0
    total_annotations: int = 0
    annotated_images: int = 0
    background_images: int = 0

    # 类别分布
    class_distribution: Dict[str, int] = field(default_factory=dict)
    images_per_class: Dict[str, int] = field(default_factory=dict)

    # 摄像头分布
    camera_distribution: Dict[str, int] = field(default_factory=dict)
    camera_bias_score: float = 100.0  # 100 = 完全均衡

    # 时间分布
    time_distribution: Dict[str, int] = field(default_factory=dict)

    # 目标尺寸分布
    object_size_bins: Dict[str, int] = field(default_factory=dict)
    small_object_ratio: float = 0.0
    medium_object_ratio: float = 0.0
    large_object_ratio: float = 0.0

    # 空标注/背景图
    background_ratio: float = 0.0

    # 偏置检测
    bias_warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dataset Intelligence Report: {self.dataset_name} {self.version}",
            f"{'='*50}",
            f"  Images:      {self.total_images} ({self.annotated_images} annotated, {self.background_images} background)",
            f"  Annotations: {self.total_annotations}",
            f"  Background:  {self.background_ratio:.1f}%",
            f"",
            f"  Object Sizes:",
            f"    Small:  {self.small_object_ratio:.1f}%",
            f"    Medium: {self.medium_object_ratio:.1f}%",
            f"    Large:  {self.large_object_ratio:.1f}%",
        ]
        if self.bias_warnings:
            lines.append(f"")
            lines.append(f"  Bias Warnings:")
            for w in self.bias_warnings:
                lines.append(f"    ! {w}")
        if self.recommendations:
            lines.append(f"")
            lines.append(f"  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    -> {r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "created_at": self.created_at or datetime.now().isoformat(),
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "annotated_images": self.annotated_images,
            "background_images": self.background_images,
            "background_ratio": self.background_ratio,
            "class_distribution": self.class_distribution,
            "camera_distribution": self.camera_distribution,
            "camera_bias_score": self.camera_bias_score,
            "time_distribution": self.time_distribution,
            "object_size_bins": self.object_size_bins,
            "small_object_ratio": self.small_object_ratio,
            "medium_object_ratio": self.medium_object_ratio,
            "large_object_ratio": self.large_object_ratio,
            "bias_warnings": self.bias_warnings,
            "recommendations": self.recommendations,
        }


class DatasetIntelligence:
    """数据智能分析器。

    分析一个 Dataset Version 下的所有数据，生成 IntelligenceReport。
    """

    def __init__(self, dataset_name: str, version: str = "v1"):
        self.dataset_name = dataset_name
        self.version = version
        self.dataset_dir = R.dataset_dir(dataset_name)
        self.version_dir = R.dataset_version_dir(dataset_name, version)
        self.manifest_path = self.version_dir / "manifest.jsonl"
        self.metadata_dir = self.dataset_dir / "metadata"

    def analyze(self) -> IntelligenceReport:
        """执行全部分析。"""
        report = IntelligenceReport(
            dataset_name=self.dataset_name,
            version=self.version,
        )

        # Step 1: Load Manifest
        manifest = self._load_manifest()
        if not manifest:
            return report

        report.total_images = len(manifest)

        # Step 2: Load annotations
        ann_dir = self.version_dir / "annotations" / "converted" / "yolo"
        annotations = {}  # image_id -> list of (class_id, w, h, area)
        if ann_dir.exists():
            for i, rec in enumerate(manifest):
                label_file = ann_dir / f"{i:06d}.txt"
                if label_file.exists() and label_file.stat().st_size > 0:
                    boxes = []
                    with open(label_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    cls_id = int(parts[0])
                                    bw = float(parts[3])
                                    bh = float(parts[4])
                                    area = bw * bh
                                    boxes.append((cls_id, bw, bh, area))
                                except ValueError:
                                    pass
                    if boxes:
                        annotations[rec.get("image_id", f"img_{i:06d}")] = boxes

        report.total_annotations = sum(len(v) for v in annotations.values())
        report.annotated_images = len(annotations)
        report.background_images = report.total_images - report.annotated_images
        report.background_ratio = (report.background_images / report.total_images * 100) if report.total_images > 0 else 0

        # Step 3: Class Distribution
        class_counter: Counter = Counter()
        for boxes in annotations.values():
            for cls_id, _, _, _ in boxes:
                class_counter[str(cls_id)] += 1
        report.class_distribution = dict(class_counter)

        # Step 4: Camera Distribution
        camera_counter: Counter = Counter()
        for rec in manifest:
            camera = rec.get("camera", rec.get("source", "unknown"))
            camera_counter[camera] += 1
        report.camera_distribution = dict(camera_counter)

        # Camera bias score
        if len(camera_counter) >= 2:
            values = list(camera_counter.values())
            max_cam = max(values)
            min_cam = min(values)
            if max_cam > 0:
                ratio = min_cam / max_cam
                report.camera_bias_score = round(ratio * 100, 1)
                if ratio < 0.2:
                    report.bias_warnings.append(
                        f"Camera imbalance: max={max_cam}, min={min_cam} (ratio={ratio:.2f})"
                    )
                    report.recommendations.append("Collect more data from underrepresented cameras")
        else:
            report.bias_warnings.append("Only one camera source — generalization may be limited")
            report.recommendations.append("Add data from more cameras to improve generalization")

        # Step 5: Time Distribution
        time_counter: Counter = Counter()
        for rec in manifest:
            ct = rec.get("capture_time", "")
            # Need at least 13 chars to extract a 2-digit hour from position 11-13.
            # '2026-07-15' (len 10) would give ct[11:13] = '' and int('') raises.
            if ct and len(ct) >= 13 and ct[11:13].isdigit():
                hour = int(ct[11:13])
                if 6 <= hour < 18:
                    time_counter["day"] += 1
                elif 18 <= hour < 21:
                    time_counter["dusk"] += 1
                elif 5 <= hour < 6:
                    time_counter["dawn"] += 1
                else:
                    time_counter["night"] += 1
            else:
                time_counter["unknown"] += 1
        report.time_distribution = dict(time_counter)

        # Check time bias
        night_ratio = time_counter.get("night", 0) / max(sum(time_counter.values()), 1) * 100
        if night_ratio < 5 and sum(time_counter.values()) > 100:
            report.bias_warnings.append(f"Very low night data: {night_ratio:.1f}%")
            report.recommendations.append("Collect more night-scene data")

        # Step 6: Object Size Distribution
        small = medium = large = 0
        size_bins: Counter = Counter()
        for boxes in annotations.values():
            for _, bw, bh, area in boxes:
                if area < 0.01:  # <1% of image area
                    small += 1
                    size_bins["small (<1%)"] += 1
                elif area < 0.1:  # 1%-10%
                    medium += 1
                    size_bins["medium (1-10%)"] += 1
                else:
                    large += 1
                    size_bins["large (>10%)"] += 1
        total_size = small + medium + large
        if total_size > 0:
            report.small_object_ratio = small / total_size * 100
            report.medium_object_ratio = medium / total_size * 100
            report.large_object_ratio = large / total_size * 100
        report.object_size_bins = dict(size_bins)

        if report.small_object_ratio > 50:
            report.recommendations.append("High small-object ratio — consider higher resolution input or tiling")

        # Step 7: Class imbalance
        if class_counter:
            cls_values = list(class_counter.values())
            max_cls = max(cls_values)
            min_cls = min(cls_values)
            if min_cls > 0 and max_cls / min_cls > 10:
                report.bias_warnings.append(f"Class imbalance: max/min ratio = {max_cls/min_cls:.1f}")
                report.recommendations.append("Apply class weights or collect more data for minority classes")

        return report

    def _load_manifest(self) -> List[Dict]:
        if not self.manifest_path.exists():
            return []
        with open(self.manifest_path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def save_report(self, report: IntelligenceReport, name: str = "latest") -> Path:
        """保存分析报告到 datasets/<name>/reports/。"""
        reports_dir = self.dataset_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"intelligence_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path
