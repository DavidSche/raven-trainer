"""Misdata Management — 误报数据管理。

生产环境中，模型部署后会持续产生误报（FP）。
误报数据是最宝贵的训练数据来源之一。

流程：
  1. 收集误报图片（来自生产环境）
  2. 分析误报（用当前模型验证、统计类别分布）
  3. 添加到数据资产（创建新版本）
  4. 重新训练

Usage:
    from raven_trainer.data_asset.misdata import MisdataManager
    mm = MisdataManager(dataset_name="workshop")
    mm.collect("/path/to/misdata/images")
    mm.analyze("/path/to/pretrained.pt")
    mm.merge(version="v2")
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from ...config import R
from ...asset_manager import AssetManager


class MisdataManager:
    """误报数据管理器。

    管理误报数据的收集、分析和入库。
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.asset_dir = R.dataset_dir(dataset_name)
        self.misdata_dir = self.asset_dir / "misdata"
        self.misdata_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, source_dir: str | Path, copy: bool = True) -> int:
        """收集误报图片到数据资产的 misdata/ 目录。

        Args:
            source_dir: 误报图片所在目录
            copy: True=复制, False=硬链接

        Returns:
            收集的图片数
        """
        src = Path(source_dir)
        count = 0
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in src.iterdir():
            if f.suffix.lower() in exts:
                dst = self.misdata_dir / f.name
                if not dst.exists():
                    if copy:
                        shutil.copy2(str(f), str(dst))
                    else:
                        try:
                            import os as _os
                            _os.link(str(f), str(dst))
                        except OSError:
                            shutil.copy2(str(f), str(dst))
                count += 1
        return count

    def analyze(self, model_path: str | Path, conf_threshold: float = 0.25) -> Dict:
        """用指定模型分析误报数据。

        Args:
            model_path: YOLO 模型路径 (.pt)
            conf_threshold: 置信度阈值

        Returns:
            分析结果统计
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"error": "ultralytics not installed"}

        model = YOLO(str(model_path))
        total = 0
        false_positives = 0
        stats: Dict[str, Any] = {"by_class": {}, "total_images": 0}

        for f in sorted(self.misdata_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            results = model(str(f), conf=conf_threshold, verbose=False)
            dets = 0
            for r in results:
                if r.boxes is not None:
                    dets += len(r.boxes)
                    for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
                        label = model.names[int(cls_id)]
                        stats["by_class"][label] = stats["by_class"].get(label, 0) + 1
            if dets > 0:
                false_positives += 1  # 所有检测在误报数据中都算 FP
            total += 1

        stats["total_images"] = total
        stats["images_with_detections"] = false_positives
        stats["total_detections"] = sum(stats["by_class"].values())
        return stats

    def merge(self, version: str = "") -> int:
        """将误报数据合并到数据集资产的新版本中。

        将 misdata/ 中的图片以 Manifest 记录形式添加到新版本。

        Returns:
            合并的图片数
        """
        from ..version import VersionManager, VersionSnapshot
        from ..registry import AssetStore, DatasetRegistry

        vm = VersionManager(self.asset_dir)
        store = AssetStore(R.DATASETS)
        registry = DatasetRegistry(store)
        asset = registry.get(self.dataset_name)

        if not asset:
            raise ValueError(f"Dataset '{self.dataset_name}' not found")

        # 创建 Manifest 记录
        am = AssetManager()
        records: List[Dict[str, Any]] = []
        for f in sorted(self.misdata_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rel = f.relative_to(R.PROJECT_ROOT)
            record = am.create_manifest_record(
                storage_path=str(rel.as_posix()),
                source="misdata",
                image_id=f"misdata_{len(records):04d}",
            )
            records.append(record)

        if not records:
            return 0

        # 创建版本
        version_str = version or f"v{len(vm.list_versions()) + 1}"
        version_dir = R.dataset_version_dir(self.dataset_name, version_str)
        am.save_manifest(records, version_dir / "manifest.jsonl")

        snap = VersionSnapshot(
            version=version_str,
            parent_version=asset.current_version or None,
            description=f"Merged {len(records)} misdata images",
            image_count=len(records),
            source="misdata",
        )
        snap.save(version_dir / "snapshot.yaml")

        # 更新 Registry
        asset.current_version = version_str
        asset.version_count = len(vm.list_versions())
        asset.total_images += len(records)
        registry.update(asset)

        return len(records)
