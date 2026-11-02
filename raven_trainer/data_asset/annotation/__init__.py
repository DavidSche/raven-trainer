"""AnnotationManager — 标注统一管理器。

定位与边界：
  Raven Trainer 的 AnnotationManager 管理本地标注文件的版本、格式、质量。
  它不与 CVAT/Label Studio 直接交互（那是 Raven-Go 的职责）。
  它管理的是已经导出到本地的标注文件。

职责：
  1. 标注格式转换（YOLO ↔ COCO ↔ CVAT XML）
  2. 标注版本管理（与 Dataset Version 绑定）
  3. 标注质量检查（空标注、无效框、类别越界）
  4. 标注统计（类别分布、目标尺寸分布）
  5. 标注文件生命周期（raw/ 真值 ↔ converted/ 构建产物）

存储结构：
  data/datasets/<name>/versions/v1/annotations/
    raw/              ← 真值（原始格式，从标注平台导出）
      cvat.xml
      labelstudio.json
    converted/        ← 构建产物（按训练框架转换）
      yolo/
      coco/

Design:
  AnnotationManager
    ├── ConverterPlugin   ← 格式转换插件（YOLO/COCO/CVATXML）
    ├── QualityChecker    ← 标注质量检查
    └── Statistics        ← 标注统计

  ConverterPlugin 是插件接口，可通过 entry_points 扩展。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config import R
from .task import AnnotationTask, TaskManager


class AnnotationManager:
    """标注管理器。

    管理一个 Dataset Version 下的所有标注文件。
    """

    def __init__(self, dataset_name: str, version: str = "v1"):
        self.dataset_name = dataset_name
        self.version = version
        self.ann_dir = R.dataset_version_dir(dataset_name, version) / "annotations"
        self.raw_dir = self.ann_dir / "raw"
        self.converted_dir = self.ann_dir / "converted"

    def list_annotations(self, format: str = "yolo") -> List[Path]:
        """列出指定格式的标注文件。"""
        target_dir = self.converted_dir / format
        if not target_dir.exists():
            return []
        return sorted(target_dir.glob("*.txt")) if format == "yolo" else sorted(target_dir.glob("*.json"))

    def get_statistics(self, format: str = "yolo") -> Dict[str, Any]:
        """统计标注信息：类别分布、目标数量、空标注数。"""
        stats: Dict[str, Any] = {
            "total_images": 0,
            "annotated_images": 0,
            "empty_images": 0,
            "total_annotations": 0,
            "class_distribution": {},
            "images_per_class": {},
        }

        label_files = self.list_annotations(format)
        if not label_files:
            return stats

        stats["total_images"] = len(label_files)
        class_counter: Counter = Counter()
        image_classes: Dict[str, set] = {}

        for lf in label_files:
            if lf.stat().st_size == 0:
                stats["empty_images"] += 1
                continue

            stats["annotated_images"] += 1
            classes_in_image = set()
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            class_counter[cls_id] += 1
                            classes_in_image.add(cls_id)
                        except ValueError:
                            pass
            for c in classes_in_image:
                image_classes.setdefault(str(c), set()).add(lf.stem)

        stats["total_annotations"] = sum(class_counter.values())
        stats["class_distribution"] = dict(class_counter)
        stats["images_per_class"] = {k: len(v) for k, v in image_classes.items()}

        return stats

    def validate(self, format: str = "yolo") -> List[Dict[str, Any]]:
        """验证标注文件质量。

        Returns:
            List of issues found:
            [{"file": "000001.txt", "issue": "empty_label", "severity": "warn"}, ...]
        """
        issues: List[Dict[str, Any]] = []
        label_files = self.list_annotations(format)
        if not label_files:
            return issues

        for lf in label_files:
            if lf.stat().st_size == 0:
                issues.append({"file": lf.name, "issue": "empty_label", "severity": "warn"})
                continue

            line_no = 0
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    line_no += 1
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        issues.append({"file": lf.name, "line": line_no, "issue": "insufficient_fields", "severity": "error"})
                        continue
                    try:
                        cls_id = int(parts[0])
                        if cls_id < 0:
                            issues.append({"file": lf.name, "line": line_no, "issue": "negative_class_id", "severity": "error"})
                        xc, yc, w, h = map(float, parts[1:5])
                        if w <= 0 or h <= 0:
                            issues.append({"file": lf.name, "line": line_no, "issue": "non_positive_dimensions", "severity": "error"})
                        if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                            issues.append({"file": lf.name, "line": line_no, "issue": "center_out_of_range", "severity": "warn"})
                    except ValueError:
                        issues.append({"file": lf.name, "line": line_no, "issue": "parse_error", "severity": "error"})

        return issues

    def convert(self, source_format: str, target_format: str, **kwargs) -> int:
        """在格式之间转换标注。

        Args:
            source_format: "yolo" | "coco" | "cvat"
            target_format: "yolo" | "coco" | "cvat"

        Returns:
            Number of annotations converted
        """
        converter_key = f"{source_format}_to_{target_format}"
        converters = {
            "yolo_to_cvat": self._yolo_to_cvat_xml,
            "cvat_to_yolo": self._cvat_to_yolo,
        }
        converter = converters.get(converter_key)
        if converter is None:
            raise ValueError(f"Unsupported conversion: {source_format} → {target_format}")
        return converter(**kwargs)  # type: ignore[operator]

    def _yolo_to_cvat_xml(self, image_width: int = 1920, image_height: int = 1080, label_map: Optional[Dict[int, str]] = None) -> int:
        """YOLO TXT → CVAT XML。"""
        from ...datasets.converters.cvat import CvatExporter

        yolo_dir = self.converted_dir / "yolo"
        if not yolo_dir.exists():
            return 0

        # Build image paths from manifest
        manifest_path = R.dataset_manifest_path(self.dataset_name, self.version)
        image_paths = []
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = R.try_resolve_storage_path(rec.get("storage_path", ""))
                    if p is not None and p.exists():
                        image_paths.append(p)

        exporter = CvatExporter(label_map or {})
        output_xml = self.raw_dir / "cvat.xml"
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        exporter.yolo_to_cvat_xml(image_paths, yolo_dir, output_xml, image_width, image_height)
        return len(image_paths)

    def _cvat_to_yolo(self, label_map: Optional[Dict[str, int]] = None) -> int:
        """CVAT XML → YOLO TXT。"""
        from ...datasets.converters.cvat import CvatImporter

        cvat_xml = self.raw_dir / "cvat.xml"
        if not cvat_xml.exists():
            return 0

        yolo_dir = self.converted_dir / "yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)

        importer = CvatImporter(label_map or {})
        # Find image dir from manifest
        manifest_path = R.dataset_manifest_path(self.dataset_name, self.version)
        image_dir = None
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = R.try_resolve_storage_path(rec.get("storage_path", ""))
                    if p is not None and p.exists():
                        image_dir = p.parent
                        break

        if image_dir is None:
            return 0

        result = importer.cvat_xml_to_yolo(cvat_xml, image_dir, yolo_dir)
        return len(result)
