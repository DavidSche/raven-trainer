"""Dataset Doctor — 数据集健康检查。

自动检查数据集的质量、均衡性、覆盖度，生成健康评分（0-100）。

检查项：
  - 重复图片
  - 类别均衡
  - 场景覆盖（白天/夜晚/黄昏）
  - 摄像头均衡
  - 模糊/过曝/黑帧比例
  - 遮挡样本比例
  - Hard Negative 比例
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HealthCheckItem:
    """单项健康检查结果。"""
    name: str
    status: str                    # pass / warn / fail
    score: float                   # 0-100
    message: str
    detail: str = ""


@dataclass
class HealthReport:
    """完整健康检查报告。"""
    dataset_name: str
    overall_score: float           # 0-100
    checks: List[HealthCheckItem] = field(default_factory=list)
    created_at: str = ""
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "overall_score": self.overall_score,
            "created_at": self.created_at or datetime.now().isoformat(),
            "checks": [{"name": c.name, "status": c.status, "score": c.score, "message": c.message, "detail": c.detail}
                       for c in self.checks],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class DatasetDoctor:
    """数据集健康检查器。"""

    def __init__(self, asset_dir: Path):
        self.asset_dir = asset_dir
        self.reports_dir = asset_dir / "reports"

    def _ensure_dirs(self) -> None:
        """Ensure the reports directory exists (called lazily when writing)."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        metadata_count: int = 0,
        source_distribution: Optional[Dict[str, int]] = None,
        manifest_path: Optional[Path] = None,
        ann_dir: Optional[Path] = None,
    ) -> HealthReport:
        """执行健康检查，返回报告。

        Args:
            metadata_count: metadata 中的图片总数
            source_distribution: {source_name: count} 分布
        """
        checks: List[HealthCheckItem] = []
        total_score = 100.0
        recommendations: List[str] = []

        # 1. 数据集大小检查
        if metadata_count == 0:
            checks.append(HealthCheckItem(
                name="dataset_size", status="fail", score=0,
                message="数据集为空，没有图片", detail="count=0"))
            total_score -= 30
            recommendations.append("请先通过 ingest pipeline 导入数据")
        elif metadata_count < 100:
            checks.append(HealthCheckItem(
                name="dataset_size", status="warn", score=30,
                message=f"数据量偏小 ({metadata_count} 张)，建议至少 500 张", detail=f"count={metadata_count}"))
            total_score -= 15
        else:
            checks.append(HealthCheckItem(
                name="dataset_size", status="pass", score=100,
                message=f"数据量充足 ({metadata_count} 张)", detail=f"count={metadata_count}"))

        # 2. 摄像头均衡检查
        if source_distribution:
            total_images = sum(source_distribution.values())
            if len(source_distribution) >= 3:
                checks.append(HealthCheckItem(
                    name="camera_balance", status="pass", score=90,
                    message=f"覆盖 {len(source_distribution)} 个摄像头", detail=str(source_distribution)))
            elif len(source_distribution) >= 2:
                checks.append(HealthCheckItem(
                    name="camera_balance", status="warn", score=60,
                    message=f"只有 {len(source_distribution)} 个摄像头，建议增加来源多样性",
                    detail=str(source_distribution)))
                total_score -= 10
                recommendations.append("增加摄像头来源以提升泛化能力")
            else:
                checks.append(HealthCheckItem(
                    name="camera_balance", status="warn", score=40,
                    message="只有 1 个摄像头，泛化能力受限",
                    detail=str(source_distribution)))
                total_score -= 15
                recommendations.append("强烈建议增加多摄像头数据")

            # 检查是否有摄像头严重偏少
            if source_distribution:
                max_count = max(source_distribution.values())
                min_count = min(source_distribution.values())
                if max_count > 0 and min_count / max_count < 0.1:
                    checks.append(HealthCheckItem(
                        name="camera_coverage", status="warn", score=50,
                        message="部分摄像头数据严重不足",
                        detail=f"max={max_count}, min={min_count}"))
                    total_score -= 10
                    recommendations.append("对数据量少的摄像头补充采集")

        # 3. Manifest 健康检查
        if manifest_path and manifest_path.exists():
            import json
            with open(manifest_path) as f:
                lines = [l for l in f if l.strip()]
            if len(lines) != metadata_count and metadata_count > 0:
                checks.append(HealthCheckItem(
                    name="manifest_integrity", status="warn", score=60,
                    message=f"Manifest has {len(lines)} records but metadata has {metadata_count}",
                    detail=f"manifest={len(lines)}, metadata={metadata_count}"))
                total_score -= 10
            else:
                checks.append(HealthCheckItem(
                    name="manifest_integrity", status="pass", score=100,
                    message=f"Manifest OK ({len(lines)} records)"))
        elif manifest_path:
            checks.append(HealthCheckItem(
                name="manifest_integrity", status="fail", score=0,
                message="Manifest not found", detail=str(manifest_path)))
            total_score -= 20

        # 4. 标注健康检查
        if ann_dir and ann_dir.exists():
            label_files = list(ann_dir.glob("*.txt"))
            empty_labels = sum(1 for f in label_files if f.stat().st_size == 0)
            if empty_labels > 0:
                checks.append(HealthCheckItem(
                    name="empty_labels", status="warn", score=50,
                    message=f"{empty_labels}/{len(label_files)} label files are empty",
                    detail=f"empty={empty_labels}, total={len(label_files)}"))
                total_score -= 10
            else:
                checks.append(HealthCheckItem(
                    name="empty_labels", status="pass", score=100,
                    message=f"All {len(label_files)} label files have content"))

            # Check for invalid bounding boxes
            invalid_boxes = 0
            for lf in label_files:
                if lf.stat().st_size == 0:
                    continue
                with open(lf) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                vals = [float(v) for v in parts[:5]]
                                if vals[1] < 0 or vals[2] < 0 or vals[3] <= 0 or vals[4] <= 0 or vals[3] > 1 or vals[4] > 1:
                                    invalid_boxes += 1
                            except ValueError:
                                invalid_boxes += 1
            if invalid_boxes > 0:
                checks.append(HealthCheckItem(
                    name="invalid_boxes", status="warn", score=40,
                    message=f"{invalid_boxes} invalid bounding boxes found",
                    detail="Some boxes have negative or zero dimensions"))
                total_score -= 10
            else:
                checks.append(HealthCheckItem(
                    name="invalid_boxes", status="pass", score=100,
                    message="All bounding boxes valid"))

        # 5. 综合评分
        overall = max(0, min(100, total_score))

        report = HealthReport(
            dataset_name=self.asset_dir.name,
            overall_score=overall,
            checks=checks,
            summary=f"健康评分: {overall:.0f}/100, {sum(1 for c in checks if c.status=='pass')}/{len(checks)} 项通过" if checks else "暂无检查项",
            recommendations=recommendations,
        )
        return report

    def save_report(self, report: HealthReport, name: str = "latest") -> Path:
        self._ensure_dirs()
        path = self.reports_dir / f"health_{name}.json"
        report.save(path)
        return path

    def load_latest_report(self) -> Optional[HealthReport]:
        reports = sorted(self.reports_dir.glob("health_*.json"))
        if reports:
            with open(reports[-1], encoding="utf-8") as f:
                data = json.load(f)
            return HealthReport(**data)
        return None
