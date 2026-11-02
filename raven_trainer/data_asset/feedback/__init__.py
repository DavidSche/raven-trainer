"""FeedbackManager — 反馈数据管理器（替代 MisdataManager）。

反馈来源：
  false_positive   误报（模型检测到的错误目标）
  false_negative   漏报（模型未检测到的目标）
  hard_case        困难样本（模型置信度低）
  customer_case    客户提供的补充数据
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional
from ...config import R
from ...asset_manager import AssetManager

FEEDBACK_CATEGORIES = ["false_positive", "false_negative", "hard_case", "customer_case"]

class FeedbackManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.asset_dir = R.dataset_dir(dataset_name)
        self.feedback_dir = self.asset_dir / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        for cat in FEEDBACK_CATEGORIES:
            (self.feedback_dir / cat).mkdir(parents=True, exist_ok=True)

    def collect(self, source_dir, category="false_positive", copy=True):
        if category not in FEEDBACK_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'")
        src = Path(source_dir)
        cat_dir = self.feedback_dir / category
        count = 0
        for f in src.iterdir():
            if f.suffix.lower() in (".jpg",".jpeg",".png"):
                dst = cat_dir / f.name
                if not dst.exists():
                    if copy: shutil.copy2(str(f), str(dst))
                    else:
                        try: import os; os.link(str(f), str(dst))
                        except: shutil.copy2(str(f), str(dst))
                count += 1
        return count

    def analyze(self, model_path, conf_threshold=0.25):
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"error": "ultralytics not installed"}
        model = YOLO(str(model_path))
        stats = {"by_category": {}, "total_images": 0, "total_detections": 0}
        for cat in FEEDBACK_CATEGORIES:
            cat_dir = self.feedback_dir / cat
            if not cat_dir.exists(): continue
            dets = 0
            for f in sorted(cat_dir.iterdir()):
                if f.suffix.lower() not in (".jpg",".jpeg",".png"): continue
                results = model(str(f), conf=conf_threshold, verbose=False)
                for r in results:
                    if r.boxes is not None: dets += len(r.boxes)
                stats["total_images"] += 1
            stats["by_category"][cat] = dets
            stats["total_detections"] += dets
        return stats

    def list_feedback(self, category=None):
        results = []
        cats = [category] if category else FEEDBACK_CATEGORIES
        for cat in cats:
            cat_dir = self.feedback_dir / cat
            if not cat_dir.exists(): continue
            reviewed = cat_dir / "_reviewed"
            approved = cat_dir / "_approved"
            for f in sorted(cat_dir.iterdir()):
                if f.name.startswith("_"): continue
                status = "pending"
                if (approved / f.name).exists(): status = "approved"
                elif (reviewed / f.name).exists(): status = "rejected"
                results.append({"filename": f.name, "category": cat, "status": status, "path": str(f)})
        return results

    def approve(self, category, filename):
        (self.feedback_dir / category / "_approved").mkdir(exist_ok=True)
        ((self.feedback_dir / category / "_approved") / filename).write_text("")

    def reject(self, category, filename):
        (self.feedback_dir / category / "_reviewed").mkdir(exist_ok=True)
        ((self.feedback_dir / category / "_reviewed") / filename).write_text("")

    def merge(self, version=""):
        from ..version import VersionManager, VersionSnapshot
        from ..registry import AssetStore, DatasetRegistry
        vm = VersionManager(self.asset_dir)
        store = AssetStore(R.DATASETS)
        registry = DatasetRegistry(store)
        asset = registry.get(self.dataset_name)
        if not asset: raise ValueError(f"Dataset '{self.dataset_name}' not found")
        am = AssetManager()
        records = []
        for cat in FEEDBACK_CATEGORIES:
            cat_dir = self.feedback_dir / cat
            approved = cat_dir / "_approved"
            for f in sorted(cat_dir.iterdir()):
                if f.name.startswith("_"): continue
                if approved.exists() and (approved / f.name).exists():
                    rel = f.relative_to(R.PROJECT_ROOT)
                    records.append(am.create_manifest_record(storage_path=str(rel.as_posix()), source=f"feedback_{cat}"))
        if not records: return 0
        version_str = version or f"v{len(vm.list_versions()) + 1}"
        am.save_manifest(records, R.dataset_version_dir(self.dataset_name, version_str) / "manifest.jsonl")
        snap = VersionSnapshot(version=version_str, parent_version=asset.current_version or None, image_count=len(records), source="feedback")
        snap.save(R.dataset_version_dir(self.dataset_name, version_str) / "snapshot.yaml")
        asset.current_version = version_str
        asset.version_count = len(vm.list_versions())
        asset.total_images += len(records)
        registry.update(asset)
        return len(records)

