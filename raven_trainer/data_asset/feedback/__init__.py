"""FeedbackManager — 反馈数据管理器（替代 MisdataManager）。

反馈来源：
  false_positive   误报（模型检测到的错误目标）
  false_negative   漏报（模型未检测到的目标）
  hard_case        困难样本（模型置信度低）
  customer_case    客户提供的补充数据
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from ...config import R, validate_name
from ...asset_manager import AssetManager

FEEDBACK_CATEGORIES = ["false_positive", "false_negative", "hard_case", "customer_case"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class FeedbackManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = validate_name(dataset_name, "dataset name")
        self.asset_dir = R.dataset_dir(self.dataset_name)
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
            if f.suffix.lower() in IMAGE_SUFFIXES:
                dst = cat_dir / f.name
                if not dst.exists():
                    if copy:
                        shutil.copy2(str(f), str(dst))
                    else:
                        try:
                            os.link(str(f), str(dst))
                        except OSError:
                            shutil.copy2(str(f), str(dst))
                    # Only count files that were actually collected; previously
                    # already-present files inflated the reported total.
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

    def _marker(self, category: str, filename: str) -> Path:
        """Path of the feedback image inside a category directory."""
        if category not in FEEDBACK_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'")
        return self.feedback_dir / category / filename

    def approve(self, category, filename):
        """Mark a feedback image as approved (required for ``merge``)."""
        item = self._marker(category, filename)
        if not item.is_file():
            raise FileNotFoundError(f"Feedback item not found: {category}/{filename}")
        marker_dir = item.parent / "_approved"
        marker_dir.mkdir(exist_ok=True)
        (marker_dir / filename).write_text("", encoding="utf-8")

    def reject(self, category, filename):
        """Mark a feedback image as rejected."""
        item = self._marker(category, filename)
        if not item.is_file():
            raise FileNotFoundError(f"Feedback item not found: {category}/{filename}")
        marker_dir = item.parent / "_reviewed"
        marker_dir.mkdir(exist_ok=True)
        (marker_dir / filename).write_text("", encoding="utf-8")

    def _current_records(self) -> List[dict]:
        """Load the manifest of the newest version (versions are cumulative)."""
        import json

        from ..version import VersionManager

        vm = VersionManager(self.asset_dir)
        versions = vm.list_versions()
        if not versions:
            return []
        manifest = vm.get_version_dir(versions[0].version) / "manifest.jsonl"
        if not manifest.exists():
            return []
        records: List[dict] = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
        return records

    def merge(self, version=""):
        """Create a new dataset version from approved feedback images.

        Versions are cumulative snapshots: the current version's manifest is
        carried over and approved feedback images are appended. Records are
        de-duplicated by ``storage_path`` so merging twice is idempotent.
        """
        from ..version import VersionManager
        from ..registry import AssetStore, DatasetRegistry

        vm = VersionManager(self.asset_dir)
        store = AssetStore(R.DATASETS)
        registry = DatasetRegistry(store)
        asset = registry.get(self.dataset_name)
        if not asset:
            raise ValueError(f"Dataset '{self.dataset_name}' not found")
        am = AssetManager()

        records = self._current_records()
        seen = {rec.get("storage_path", "") for rec in records}
        added = 0
        for cat in FEEDBACK_CATEGORIES:
            cat_dir = self.feedback_dir / cat
            approved = cat_dir / "_approved"
            if not cat_dir.exists():
                continue
            for f in sorted(cat_dir.iterdir()):
                if f.name.startswith("_"):
                    continue
                if not (approved.exists() and (approved / f.name).exists()):
                    continue
                try:
                    storage_path = f.resolve().relative_to(R.PROJECT_ROOT).as_posix()
                except ValueError:
                    # Custom RAVEN_DATA root outside the project: store an
                    # absolute path instead of crashing the merge.
                    storage_path = str(f.resolve())
                if storage_path in seen:
                    continue
                seen.add(storage_path)
                records.append(
                    am.create_manifest_record(
                        storage_path=storage_path, source=f"feedback_{cat}"
                    )
                )
                added += 1
        if added == 0:
            return 0
        snapshot = vm.create_version(
            manifest_records=records,
            description=f"Feedback merge (+{added} images)",
            source="feedback",
            version_tag=version,
        )
        asset.current_version = snapshot.version
        asset.version_count = len(vm.list_versions())
        asset.total_images = snapshot.image_count
        registry.update(asset)
        return added
