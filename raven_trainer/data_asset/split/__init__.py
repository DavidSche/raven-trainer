"""Dataset Split — 数据集划分工具。

支持摄像头感知的分层抽样：
  1. 识别每张图片的摄像头来源（从文件名或 Manifest metadata 中提取）
  2. 按摄像头分组，保证各摄像头在 train/val 中分布比例一致
  3. 在每个摄像头内部，保证单目标/多目标/背景图分布一致
  4. 输出 YOLO 格式的 train/val/test 目录

Usage:
    from raven_trainer.data_asset.split import DatasetSplitter
    splitter = DatasetSplitter(manifest_path, ann_dir)
    splitter.split(output_dir, val_size=0.2, test_size=0.0)
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...config import R
from ...asset_manager import AssetManager


class DatasetSplitter:
    """数据集划分器。

    在 YOLO 格式的标注数据上执行摄像头感知的分层抽样划分。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        ann_dir: str | Path,
        project_root: Optional[Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.ann_dir = Path(ann_dir)
        self.project_root = project_root or R.PROJECT_ROOT

    def _load_manifest(self) -> List[dict]:
        with open(self.manifest_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _extract_camera_id(self, filename: str, fallback_source: str = "") -> str:
        """从文件名中提取摄像头 ID。支持多种命名格式。

        格式示例:
          - ori_4_6541170175150032_20260115_153436_163.jpg → 6541170175150032
          - playback_172.101.1.247_05_20260601100232200.jpeg → 172.101.1.247
          - frame_004165.jpg → fallback_source
        """
        name = Path(filename).stem
        parts = name.split("_")

        # 尝试提取 IP 格式的摄像头 ID
        for p in parts:
            if p.count(".") == 3 and all(x.isdigit() for x in p.split(".")):
                return p

        # 新格式：ori_4_6541232880390464_6541170175150032_20260115_153436_163
        if len(parts) >= 4 and parts[0] == "ori":
            return parts[3] if len(parts) >= 7 else (parts[2] if len(parts) >= 6 else parts[1])

        # 视频帧格式：OP4-R1-03000002353000000_frame_004165
        if len(parts) >= 3 and "frame" in parts:
            return parts[0].split("-")[-1] if "-" in parts[0] else parts[0]

        return fallback_source or "unknown"

    def _count_labels(self, label_path: Path) -> int:
        if not label_path.exists():
            return 0
        with open(label_path, encoding="utf-8") as f:
            return len([l for l in f if l.strip()])

    def _classify_sample(self, label_count: int) -> str:
        if label_count == 0:
            return "background"
        elif label_count == 1:
            return "single"
        else:
            return "multi"

    def split(
        self,
        output_dir: str | Path,
        val_size: float = 0.2,
        test_size: float = 0.0,
        seed: int = 42,
        link: bool = True,
    ) -> Dict[str, int]:
        """执行数据集划分。

        Args:
            output_dir: 输出目录
            val_size: 验证集比例
            test_size: 测试集比例（剩余为训练集）
            seed: 随机种子
            link: 是否使用硬链接（否则复制）

        Returns:
            {"train": N, "val": N, "test": N}
        """
        out = Path(output_dir)
        random.seed(seed)

        # 读取 Manifest
        manifest = self._load_manifest()

        # 按摄像头分组统计
        camera_groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in manifest:
            filename = Path(rec.get("storage_path", "")).name
            source = rec.get("source", "")
            cam = self._extract_camera_id(filename, source)
            camera_groups[cam].append(rec)

        train_set: List[dict] = []
        val_set: List[dict] = []
        test_set: List[dict] = []

        for cam, samples in camera_groups.items():
            # 在每个摄像头内部，按标签数分层
            stratified: Dict[str, List[dict]] = {"background": [], "single": [], "multi": []}
            for rec in samples:
                # 查找对应标注文件
                idx = int(rec.get("image_id", "0").split("_")[1]) if "_" in rec.get("image_id", "") else 0
                label_path = self.ann_dir / f"{idx:06d}.txt"
                lc = self._count_labels(label_path)
                category = self._classify_sample(lc)
                stratified[category].append(rec)

            for category, cat_samples in stratified.items():
                if not cat_samples:
                    continue
                random.shuffle(cat_samples)
                n = len(cat_samples)
                # Match test_size semantics: only allocate val samples when
                # val_size > 0, otherwise n_val=0 (was: max(1, ...) forced ≥1).
                n_val = max(1, int(n * val_size)) if n > 1 and val_size > 0 else 0
                n_test = max(1, int(n * test_size)) if n > 1 and test_size > 0 else 0

                # 取最后 n_val 个作为 val，n_test 个作为 test，其余 train
                if n_test > 0:
                    val_set.extend(cat_samples[-n_val - n_test:-n_test])
                    test_set.extend(cat_samples[-n_test:])
                else:
                    val_set.extend(cat_samples[-n_val:])
                train_set.extend(cat_samples[:max(0, n - n_val - n_test)])

        # 创建输出目录
        dirs = {
            "train": out / "train",
            "val": out / "val",
        }
        if test_size > 0:
            dirs["test"] = out / "test"

        for split_name, split_dir in dirs.items():
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        # 写入文件
        am = AssetManager()
        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for split_name, split_samples in [("train", train_set), ("val", val_set), ("test", test_set)]:
            if split_name not in dirs:
                continue
            img_dir = dirs[split_name] / "images"
            lbl_dir = dirs[split_name] / "labels"

            for rec in split_samples:
                storage_path = rec.get("storage_path", "")
                # Let pathlib handle path separators cross-platform.
                # Was: storage_path.replace('/', '\\') which broke Linux/macOS
                # (backslash is a legal filename char there, not a separator).
                src_img = self.project_root / storage_path
                if not src_img.exists():
                    continue
                dst_img = img_dir / src_img.name
                if not dst_img.exists():
                    am.link_to_workspace([src_img], dirs[split_name])

                # 复制标注
                idx = int(rec.get("image_id", "0").split("_")[1]) if "_" in rec.get("image_id", "") else 0
                src_label = self.ann_dir / f"{idx:06d}.txt"
                if src_label.exists():
                    dst_label = lbl_dir / f"{src_img.stem}.txt"
                    shutil.copy2(str(src_label), str(dst_label))
                counts[split_name] += 1

        return counts
