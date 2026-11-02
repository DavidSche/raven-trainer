"""Metadata Extractor — 在 ingestion 中独立提取结构化元数据。

从图片文件和路径中提取：
  - 来源信息：camera_id (从路径或文件名 IP 提取)
  - 时间信息：capture_time (从文件名 YYYYMMDDHHMMSS 提取)
  - 设备信息：resolution, file_size
  - 质量信息：blur_score, brightness (调用 quality checker)

设计原则：Metadata Extract 是 Ingestion Pipeline 的独立节点，
不与 Importer 耦合。未来增加 GPS/weather/camera_model 等字段时
只需扩展此模块，不需修改 Importer。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..config import R


class MetadataExtractor:
    """从图片文件和路径中提取结构化元数据。"""

    def extract(self, image_path: Path, source: str = "") -> Dict[str, Any]:
        """提取单张图片的完整元数据。

        Args:
            image_path: 图片文件的绝对路径
            source: 来源标识

        Returns:
            dict 包含 image_id, filename, camera, capture_time, width, height,
            file_size, sha256, blur_score, brightness, quality_score
        """
        meta: Dict[str, Any] = {
            "filename": image_path.name,
            "source": source,
        }

        if not image_path.exists():
            return meta

        # 文件信息
        meta["file_size"] = image_path.stat().st_size

        # SHA256 (prefix only — first 64KB)
        meta["sha256"] = self._compute_sha256_prefix(image_path)

        # 从文件名提取时间和摄像头
        stem = image_path.stem
        time_match = re.search(r"(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})", stem)
        if time_match:
            y, mo, d, h, mi, s = time_match.groups()
            meta["capture_time"] = f"{y}-{mo}-{d}T{h}:{mi}:{s}"

        ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", stem)
        if ip_match:
            meta["camera"] = ip_match.group(1)

        # 图像信息（需要 OpenCV）
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            meta["width"] = w
            meta["height"] = h

            # 质量评分
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            meta["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            meta["brightness"] = float(np.mean(img))
            # quality_score: 0-100, 基于 blur_score 归一化
            meta["quality_score"] = min(100.0, meta["blur_score"] / 5.0)

        return meta

    def extract_batch(self, image_paths: List[Path], source: str = "") -> List[Dict[str, Any]]:
        """批量提取元数据。"""
        return [self.extract(p, source) for p in image_paths]

    def save_to_metadata_store(self, records: List[Dict[str, Any]], dataset_name: str, version: str) -> int:
        """将提取的元数据写入 data/datasets/<name>/metadata/。"""
        from ..data_asset.metadata import MetadataStore, ImageMetadata

        metas = []
        for i, rec in enumerate(records):
            metas.append(ImageMetadata(
                image_id=rec.get("image_id", f"{dataset_name}_{i:06d}"),
                filename=rec.get("filename", ""),
                camera=rec.get("camera", ""),
                width=rec.get("width", 0),
                height=rec.get("height", 0),
                file_size=rec.get("file_size", 0),
                file_hash=rec.get("sha256", ""),
                blur_score=rec.get("blur_score", 0.0),
                brightness=rec.get("brightness", 0.0),
                quality_score=rec.get("quality_score", 0.0),
            ))
        # One transaction + one connection lifetime per batch instead of
        # committing (and leaking a SQLite handle) per image.
        with MetadataStore(R.dataset_dir(dataset_name) / "metadata") as ms:
            ms.add_batch(metas)
        return len(metas)

    def _compute_sha256_prefix(self, path: Path, sample_size: int = 65536) -> str:
        """Compute SHA256 from the first ``sample_size`` bytes (default 64KB).

        This is NOT a full-file hash — it is a fast prefix hash used for
        approximate deduplication.  For full-file integrity checks, read
        the entire file instead.
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
            h.update(chunk)
        return h.hexdigest()
