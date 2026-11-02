"""Workspace Builder — 根据 Recipe + Profile 构建训练工作区。

把一个 Recipe（数据集构建规则）物化成实际的训练文件结构：

  workspace/<name>/
    dataset.yaml          ← YOLO 格式的数据集描述
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/                 (optional)
      images/
      labels/

流程：
  1. 读取 Recipe，解析 sources（引用哪些 Dataset / Version）
  2. 从每个 version 加载 images + annotations
  3. 按 Recipe 的 include/exclude/sampling 过滤
  4. 按 Profile 的 include_classes/exclude_classes/filter 过滤
  5. 跨版本全局去重（dhash）
  6. 摄像头感知分层抽样 split
  7. 转换标注格式（CVAT XML → YOLO TXT / COCO JSON）
  8. 生成 dataset.yaml
  9. 写入 lineage
"""

from __future__ import annotations

import json
import shutil
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imagehash
import yaml
from PIL import Image

from ...config import R
from ..recipe import DatasetRecipe, RecipeSource, RecipeManager, CatalogRecipeManager
from ..profile import BuildProfile
from ..version import VersionManager
from ..lineage import LineageStore, LineageNode, LineageEdge


# ──────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────


@dataclass
class WorkspaceSample:
    """A single sample in the built workspace, with resolved paths."""
    image_id: str
    image_src: Path           # source image path
    label_src: Optional[Path] = None
    camera: str = ""
    class_ids: List[int] = field(default_factory=list)
    quality_score: float = 1.0
    source_version: str = ""


@dataclass
class WorkspaceBuildResult:
    """Result of a workspace build."""
    name: str
    output_dir: Path
    dataset_yaml: Path
    total_images: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_count: int = 0
    class_names: Dict[int, str] = field(default_factory=dict)
    source_versions: List[str] = field(default_factory=list)
    skipped_blurry: int = 0
    skipped_empty: int = 0
    skipped_duplicate: int = 0
    duration_sec: float = 0.0
    created_at: str = ""


# ──────────────────────────────────────────────
#  WorkspaceBuilder
# ──────────────────────────────────────────────


class WorkspaceBuilder:
    """Dataset Workspace Builder.

    Usage:
        builder = WorkspaceBuilder(dataset_name="person_detection")
        result = builder.build(
            recipe_name="default",
            profile=BuildProfile.detection_default(),
            output_dir=Path("data/workspace/person_v3"),
        )
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: Optional[Path] = None,
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root or R.DATASETS
        self.asset_dir = self.data_root / dataset_name
        self._stats: Dict[str, int] = {
            "skipped_blurry": 0,
            "skipped_empty": 0,
            "skipped_duplicate": 0,
        }

    # ── Public API ──

    def build(
        self,
        recipe_name: str = "default",
        profile: Optional[BuildProfile] = None,
        output_dir: Optional[Path] = None,
        force: bool = False,
        seed: int = 42,
    ) -> WorkspaceBuildResult:
        """Build a training workspace from Recipe + Profile.

        Args:
            recipe_name: Name of the recipe to use (e.g. "default", "v3")
            profile: BuildProfile config; uses detection_default() if not given
            output_dir: Output directory; defaults to data/workspace/<dataset_name>/
            force: Overwrite existing output directory
            seed: Random seed for split reproducibility

        Returns:
            WorkspaceBuildResult with counts and paths
        """
        import time as _time
        t0 = _time.time()

        # 1. Load recipe
        rm = RecipeManager(self.asset_dir)
        recipe = rm.get_recipe(recipe_name)
        if recipe is None:
            raise FileNotFoundError(
                f"Recipe '{recipe_name}' not found for dataset '{self.dataset_name}'. "
                f"Use 'raven-train dataset recipe {self.dataset_name} --create' first."
            )

        # 2. Load or default profile
        if profile is None:
            profile = BuildProfile.detection_default()

        # 3. Determine output directory
        if output_dir is None:
            output_dir = R.workspace_dir(f"{self.dataset_name}_{recipe_name}")
        out = Path(output_dir)
        if out.exists():
            if force:
                shutil.rmtree(out)
                out.mkdir(parents=True, exist_ok=True)
            else:
                raise FileExistsError(
                    f"Workspace already exists: {out}. Use force=True to overwrite."
                )
        out.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        self._stats = {"skipped_blurry": 0, "skipped_empty": 0, "skipped_duplicate": 0}

        # 4. Resolve sources → collect samples
        print(f"[WorkspaceBuilder] Resolving sources from recipe '{recipe_name}'...")
        all_samples: List[WorkspaceSample] = []
        class_names_global: Dict[int, str] = {}
        source_versions: List[str] = []

        for src in recipe.sources:
            samples, names = self._resolve_source(src, recipe)
            all_samples.extend(samples)
            class_names_global.update(names)
            if src.name not in source_versions:
                source_versions.append(src.name)

        if not all_samples:
            raise RuntimeError("No samples resolved from recipe sources.")

        print(f"  Resolved {len(all_samples)} raw samples from {len(source_versions)} source(s)")

        # 5. Apply filtering
        print("[WorkspaceBuilder] Applying filters...")
        filtered = self._apply_filters(all_samples, recipe, profile)
        print(f"  After filter: {len(filtered)} samples "
              f"(blurry={self._stats['skipped_blurry']}, "
              f"empty={self._stats['skipped_empty']})")

        # 6. Global inter-version dedup
        print("[WorkspaceBuilder] Cross-version dedup (dhash)...")
        deduped = self._dedup_samples(filtered)
        print(f"  After dedup: {len(deduped)} samples (removed {self._stats['skipped_duplicate']})")

        # 7. Camera-aware stratified split
        print("[WorkspaceBuilder] Splitting...")
        train_set, val_set, test_set = self._split_samples(
            deduped, recipe.split, seed,
        )

        # 8. Write workspace files + dataset.yaml
        print("[WorkspaceBuilder] Writing workspace files...")
        dataset_yaml_path = out / "dataset.yaml"
        counts = self._write_workspace(
            out, train_set, val_set, test_set,
            class_names_global, profile, dataset_yaml_path,
        )

        # 9. Record lineage
        self._write_lineage(source_versions, recipe_name, str(out))

        elapsed = _time.time() - t0
        result = WorkspaceBuildResult(
            name=f"{self.dataset_name}/{recipe_name}",
            output_dir=out,
            dataset_yaml=dataset_yaml_path,
            total_images=len(deduped),
            train_count=counts["train"],
            val_count=counts["val"],
            test_count=counts["test"],
            class_count=len(class_names_global),
            class_names=class_names_global,
            source_versions=source_versions,
            skipped_blurry=self._stats["skipped_blurry"],
            skipped_empty=self._stats["skipped_empty"],
            skipped_duplicate=self._stats["skipped_duplicate"],
            duration_sec=round(elapsed, 1),
            created_at=datetime.now().isoformat(),
        )

        print(f"\n[WorkspaceBuilder] Done in {result.duration_sec}s")
        print(f"  Train: {result.train_count}  Val: {result.val_count}  Test: {result.test_count}")
        print(f"  Classes: {result.class_count}")
        print(f"  dataset.yaml: {result.dataset_yaml}")
        print(f"  Output: {result.output_dir}")

        return result

    def build_from_catalog(
        self,
        recipe_name: str,
        profile: Optional[BuildProfile] = None,
        output_dir: Optional[Path] = None,
        force: bool = False,
        seed: int = 42,
    ) -> WorkspaceBuildResult:
        """Build workspace from a standalone catalog recipe (data/recipes/<name>/).

        This is the recommended entry point for the new standalone Recipe system.
        Unlike build(), this does not require pre-registered dataset-level recipes.
        """
        mgr = CatalogRecipeManager()
        recipe = mgr.get_recipe(recipe_name)
        if recipe is None:
            raise FileNotFoundError(
                f"Recipe '{recipe_name}' not found in catalog ({mgr.recipes_root}). "
                f"Available: {mgr.list_recipes()}"
            )
        self.dataset_name = recipe.name
        if profile is None:
            profile = BuildProfile.detection_default()
        return self._build_with_recipe(recipe, profile, output_dir, force, seed)

    # ── Shared build logic ──

    def _build_with_recipe(
        self,
        recipe: DatasetRecipe,
        profile: BuildProfile,
        output_dir: Optional[Path],
        force: bool,
        seed: int,
    ) -> WorkspaceBuildResult:
        """Shared build logic used by both build() and build_from_catalog()."""
        import time as _time
        t0 = _time.time()

        if output_dir is None:
            output_dir = R.workspace_dir(recipe.name)
        out = Path(output_dir)
        if out.exists():
            if force:
                shutil.rmtree(out)
                out.mkdir(parents=True, exist_ok=True)
            else:
                raise FileExistsError(f"Workspace already exists: {out}. Use force=True.")
        out.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        self._stats = {"skipped_blurry": 0, "skipped_empty": 0, "skipped_duplicate": 0}

        print(f"[WorkspaceBuilder] Resolving sources from recipe '{recipe.name}'...")
        all_samples, class_names_global, source_versions = self._resolve_all_sources(recipe)

        filtered = self._apply_filters(all_samples, recipe, profile)
        deduped = self._dedup_samples(filtered)
        train_set, val_set, test_set = self._split_samples(deduped, recipe.split, seed)

        dataset_yaml_path = out / "dataset.yaml"
        counts = self._write_workspace(out, train_set, val_set, test_set,
                                       class_names_global, profile, dataset_yaml_path)
        self._write_lineage(source_versions, recipe.name, str(out))

        elapsed = _time.time() - t0
        result = WorkspaceBuildResult(
            name=f"{self.dataset_name}/{recipe.name}",
            output_dir=out, dataset_yaml=dataset_yaml_path,
            total_images=len(deduped),
            train_count=counts["train"], val_count=counts["val"], test_count=counts["test"],
            class_count=len(class_names_global), class_names=class_names_global,
            source_versions=source_versions,
            skipped_blurry=self._stats["skipped_blurry"],
            skipped_empty=self._stats["skipped_empty"],
            skipped_duplicate=self._stats["skipped_duplicate"],
            duration_sec=round(elapsed, 1), created_at=datetime.now().isoformat(),
        )
        print(f"\n[WorkspaceBuilder] Done in {result.duration_sec}s")
        print(f"  Train: {result.train_count}  Val: {result.val_count}  Test: {result.test_count}")
        return result

    def _resolve_all_sources(self, recipe):
        """Resolve all sources from a recipe. Shared by build() and build_from_catalog()."""
        all_samples = []
        class_names_global = {}
        source_versions = []
        for src in recipe.sources:
            samples, names = self._resolve_source(src, recipe)
            all_samples.extend(samples)
            class_names_global.update(names)
            if src.name not in source_versions:
                source_versions.append(src.name)
        if not all_samples:
            raise RuntimeError("No samples resolved from recipe sources.")
        print(f"  Resolved {len(all_samples)} samples from {len(source_versions)} source(s)")
        return all_samples, class_names_global, source_versions
    # ── Internal: Source resolution ──

    def _resolve_source(
        self,
        src: RecipeSource,
        recipe: DatasetRecipe,
    ) -> Tuple[List[WorkspaceSample], Dict[int, str]]:
        """Resolve a single RecipeSource into samples.

        A RecipeSource can reference:
          - A dataset version (name like "my_dataset@v2" or just "my_dataset")
          - A raw image directory (path field)
        """
        samples: List[WorkspaceSample] = []
        class_names: Dict[int, str] = {}

        # Try resolving as dataset version
        version_name = ""
        dataset_name = src.name
        if "@" in src.name:
            dataset_name, version_name = src.name.split("@", 1)

        version_dir = self.data_root / dataset_name / "versions"
        if version_dir.exists():
            vm = VersionManager(self.data_root / dataset_name)
            versions = vm.list_versions()

            if version_name:
                target = [v for v in versions if v.version == version_name]
            else:
                target = versions[:1] if versions else []

            for snap in target:
                v_dir = version_dir / snap.version
                ann_dir = v_dir / "annotations" / "converted" / "yolo"
                manifest_path = v_dir / "manifest.jsonl"

                # Load class names from asset.yaml
                asset_yaml = self.data_root / dataset_name / "asset.yaml"
                if asset_yaml.exists():
                    with open(asset_yaml, encoding="utf-8") as f:
                        ad = yaml.safe_load(f)
                        if ad and isinstance(ad, dict) and "names" in ad:
                            names_raw = ad["names"]
                            if isinstance(names_raw, dict):
                                class_names.update({int(k) if not isinstance(k, int) else k: v
                                                     for k, v in names_raw.items()})

                # Load samples from manifest
                manifest_records = self._load_manifest(manifest_path)
                for i, rec in enumerate(manifest_records):
                    img_path = self._resolve_image_path(rec)
                    if img_path is None or not img_path.exists():
                        continue

                    # Corresponding label
                    img_stem = img_path.stem
                    label_path = ann_dir / f"{img_stem}.txt" if ann_dir.exists() else None
                    if label_path and not label_path.exists():
                        label_path = ann_dir / f"{i:06d}.txt"
                        if not label_path.exists():
                            label_path = None

                    # Extract class IDs from label
                    cls_ids = []
                    if label_path and label_path.exists():
                        with open(label_path, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    cls_ids.append(int(line.split()[0]))

                    cam = rec.get("camera", "") or rec.get("source", "")
                    qs = float(rec.get("quality_score", 1.0))

                    sample = WorkspaceSample(
                        image_id=rec.get("image_id", f"{img_stem}"),
                        image_src=img_path,
                        label_src=label_path,
                        camera=cam,
                        class_ids=cls_ids,
                        quality_score=qs,
                        source_version=snap.version,
                    )
                    samples.append(sample)

        # If no dataset version found, try raw path
        if not samples and src.path:
            p = Path(src.path)
            if p.exists() and p.is_dir():
                for img_file in sorted(p.iterdir()):
                    if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                        continue
                    sample = WorkspaceSample(
                        image_id=img_file.stem,
                        image_src=img_file,
                        camera=src.name,
                        source_version="raw",
                    )
                    samples.append(sample)

        return samples, class_names

    def _load_manifest(self, path: Path) -> List[dict]:
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _resolve_image_path(self, rec: dict) -> Optional[Path]:
        storage_path = rec.get("storage_path", "")
        if not storage_path:
            return None
        p = Path(storage_path)
        if p.is_absolute():
            return p if p.exists() else None
        candidate = self.data_root.parent / p
        if candidate.exists():
            return candidate
        candidate = R.PROJECT_ROOT / p
        if candidate.exists():
            return candidate
        return None

    # ── Internal: Filtering ──

    def _apply_filters(
        self,
        samples: List[WorkspaceSample],
        recipe: DatasetRecipe,
        profile: BuildProfile,
    ) -> List[WorkspaceSample]:
        """Apply quality + class + content filters from Recipe and Profile."""
        self._stats["skipped_blurry"] = 0
        self._stats["skipped_empty"] = 0

        filtered: List[WorkspaceSample] = []
        min_blur = recipe.quality.get("min_blur_score", 0)
        max_per_camera = recipe.balance.get("max_per_camera", 0)
        camera_counts: Dict[str, int] = defaultdict(int)

        include_classes = set(profile.include_classes) if profile.include_classes else None
        exclude_classes = set(profile.exclude_classes) if profile.exclude_classes else None
        remove_empty = profile.filter.get("remove_empty_labels", True)

        for s in samples:
            # Quality filter
            if min_blur > 0 and s.quality_score > 0:
                if s.quality_score < min_blur / 100.0:
                    self._stats["skipped_blurry"] += 1
                    continue

            # Empty label filter (only if the source had a label file at all)
            if remove_empty and not s.class_ids and s.label_src is not None:
                self._stats["skipped_empty"] += 1
                continue

            # Class include/exclude
            if (include_classes or exclude_classes) and s.class_ids:
                if include_classes and not any(c in include_classes for c in s.class_ids):
                    continue
                if exclude_classes and any(c in exclude_classes for c in s.class_ids):
                    continue

            # Camera balance
            if max_per_camera > 0:
                cam = s.camera or "_unknown"
                if camera_counts[cam] >= max_per_camera:
                    continue
                camera_counts[cam] += 1

            filtered.append(s)

        # Recipe-level include/exclude filename patterns
        if recipe.include:
            filtered = [
                s for s in filtered
                if any(p in s.image_src.name for p in recipe.include)
            ]
        if recipe.exclude:
            filtered = [
                s for s in filtered
                if not any(p in s.image_src.name for p in recipe.exclude)
            ]

        return filtered

    # ── Internal: Global dedup ──

    def _dedup_samples(
        self,
        samples: List[WorkspaceSample],
        hash_distance: int = 5,
    ) -> List[WorkspaceSample]:
        """Cross-version global dedup using dhash.

        Unlike the per-run dedup in ingestion, this operates across
        all resolved versions at build time.
        """
        self._stats["skipped_duplicate"] = 0
        kept: List[WorkspaceSample] = []
        # Store pre-parsed hash objects to avoid O(N²) re-parsing and to
        # release file handles promptly. Each entry: hash_str -> (hash_obj, sample).
        seen_entries: List[Tuple[Any, WorkspaceSample]] = []

        for s in samples:
            if not s.image_src.exists():
                continue
            try:
                # Use a context manager so the PIL file handle is released
                # immediately after hashing (was: Image.open without close).
                with Image.open(s.image_src) as img:
                    h_obj = imagehash.dhash(img, hash_size=8)
            except Exception:
                kept.append(s)
                continue

            # O(N) per new sample but with parsed hash objects cached, no
            # repeat hex_to_hash parsing. For large N consider a BK-tree or
            # bucketing by the first 8 bits; current cost is acceptable for
            # typical workspace sizes (<= ~10k samples).
            duplicate = False
            for existing_h_obj, _ in seen_entries:
                if (h_obj - existing_h_obj) <= hash_distance:
                    duplicate = True
                    break

            if duplicate:
                self._stats["skipped_duplicate"] += 1
            else:
                seen_entries.append((h_obj, s))
                kept.append(s)

        return kept

    # ── Internal: Camera-aware split ──

    def _split_samples(
        self,
        samples: List[WorkspaceSample],
        split: Any,
        seed: int,
    ) -> Tuple[List[WorkspaceSample], List[WorkspaceSample], List[WorkspaceSample]]:
        """Dataset split with 5 strategies.

        Inspired by aiva-trainer:
          1. random      - Simple random shuffle
          2. stratified  - Class-balanced stratified split
          3. group       - Group-wise split (entire cameras as units, anti-leakage)

        Supplementary (backward compat):
          4. by_video    - Per-camera random split (default)
          5. temporal    - Time-ordered split by filename
        """
        rng = random.Random(seed)
        strategy = split.strategy
        train_ratio = split.train
        val_ratio = split.val
        test_ratio = split.test

        # 1. Random
        if strategy == "random":
            shuffled = list(samples)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            return (shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test])

        # 2. Stratified: class-balanced
        if strategy == "stratified":
            from collections import defaultdict as _dd
            buckets = _dd(list)
            for s in samples:
                key = tuple(sorted(set(s.class_ids))) if s.class_ids else ("_bg",)
                buckets[str(key)].append(s)
            train_set, val_set, test_set = [], [], []
            for bucket in buckets.values():
                rng.shuffle(bucket)
                n = len(bucket)
                n_val = max(1, int(n * val_ratio)) if n > 2 else 0
                n_test = max(1, int(n * test_ratio)) if n > 2 and test_ratio > 0 else 0
                if n_test > 0:
                    val_set.extend(bucket[-n_val - n_test:-n_test])
                    test_set.extend(bucket[-n_test:])
                else:
                    val_set.extend(bucket[-n_val:])
                train_set.extend(bucket[:max(0, n - n_val - n_test)])
            return train_set, val_set, test_set

        # 3. Group: entire cameras to one split (anti-leakage)
        if strategy == "group":
            from collections import defaultdict as _dd
            camera_groups = _dd(list)
            for s in samples:
                camera_groups[s.camera or "_unknown"].append(s)
            group_names = list(camera_groups.keys())
            rng.shuffle(group_names)
            n_groups = len(group_names)
            n_val_groups = max(1, int(n_groups * val_ratio))
            n_test_groups = max(1, int(n_groups * test_ratio)) if test_ratio > 0 else 0
            test_groups = set(group_names[:n_test_groups])
            val_groups = set(group_names[n_test_groups:n_test_groups + n_val_groups])
            train_groups = set(group_names[n_test_groups + n_val_groups:])
            train_set = [s for g in train_groups for s in camera_groups[g]]
            val_set = [s for g in val_groups for s in camera_groups[g]]
            test_set = [s for g in test_groups for s in camera_groups[g]]
            return train_set, val_set, test_set

        # 4. Temporal: time-ordered
        if strategy == "temporal":
            sorted_s = sorted(samples, key=lambda s: s.image_src.name)
            n = len(sorted_s)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            return (sorted_s[:n_train], sorted_s[n_train:n_train + n_val], sorted_s[n_train + n_val:])

        # 5. by_video (default): per-camera random (backward compat)
        from collections import defaultdict as _dd
        camera_groups = _dd(list)
        for s in samples:
            camera_groups[s.camera or "_unknown"].append(s)
        train_set, val_set, test_set = [], [], []
        for group in camera_groups.values():
            rng.shuffle(group)
            n = len(group)
            n_val = max(1, int(n * val_ratio)) if n > 2 else 0
            n_test = max(1, int(n * test_ratio)) if n > 2 and test_ratio > 0 else 0
            if n_test > 0:
                val_set.extend(group[-n_val - n_test:-n_test])
                test_set.extend(group[-n_test:])
            else:
                val_set.extend(group[-n_val:])
            train_set.extend(group[:max(0, n - n_val - n_test)])
        return train_set, val_set, test_set
    def _write_workspace(
        self,
        out_dir: Path,
        train_set: List[WorkspaceSample],
        val_set: List[WorkspaceSample],
        test_set: List[WorkspaceSample],
        class_names: Dict[int, str],
        profile: BuildProfile,
        dataset_yaml_path: Path,
    ) -> Dict[str, int]:
        """Write images + labels + dataset.yaml to output directory."""
        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for split_name, split_samples in [
            ("train", train_set),
            ("val", val_set),
            ("test", test_set),
        ]:
            if not split_samples:
                continue
            img_dir = out_dir / split_name / "images"
            lbl_dir = out_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for sample in split_samples:
                dst_img = img_dir / sample.image_src.name
                if not dst_img.exists():
                    try:
                        dst_img.hardlink_to(sample.image_src)
                    except (OSError, AttributeError):
                        shutil.copy2(str(sample.image_src), str(dst_img))

                if sample.label_src and sample.label_src.exists():
                    dst_lbl = lbl_dir / f"{sample.image_src.stem}.txt"
                    if not dst_lbl.exists():
                        shutil.copy2(str(sample.label_src), str(dst_lbl))
                else:
                    dst_lbl = lbl_dir / f"{sample.image_src.stem}.txt"
                    if not dst_lbl.exists():
                        dst_lbl.touch()

                counts[split_name] += 1

        # Generate dataset.yaml
        all_names: Dict[int, str] = dict(class_names)
        if not all_names:
            max_cls = 0
            for s in train_set:
                for c in s.class_ids:
                    max_cls = max(max_cls, c)
            all_names = {i: str(i) for i in range(max_cls + 1)}

        dataset_yaml = {
            "path": str(out_dir.resolve().as_posix()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(all_names),
            "names": {int(k): v for k, v in all_names.items()},
        }
        if test_set:
            dataset_yaml["test"] = "test/images"
        if profile.task:
            dataset_yaml["task"] = profile.task
        dataset_yaml["format"] = profile.target_format
        dataset_yaml["hard_negative_ratio"] = 0.0
        dataset_yaml["split_method"] = "by_video"

        with open(dataset_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

        return counts

    # ── Internal: Lineage ──

    def _write_lineage(
        self,
        source_versions: List[str],
        recipe_name: str,
        workspace_path: str,
    ) -> None:
        """Record lineage: Source Versions → Recipe → Workspace."""
        try:
            ls = LineageStore(self.asset_dir)

            for sv in source_versions:
                node_id = f"{self.dataset_name}_{sv}"
                ls.add_node(LineageNode(
                    id=node_id,
                    type="version",
                    name=f"{self.dataset_name}@{sv}",
                ))

            recipe_node_id = f"{self.dataset_name}_recipe_{recipe_name}"
            ls.add_node(LineageNode(
                id=recipe_node_id,
                type="recipe",
                name=recipe_name,
            ))
            for sv in source_versions:
                ls.add_edge(LineageEdge(
                    source_id=f"{self.dataset_name}_{sv}",
                    target_id=recipe_node_id,
                    relation="groups",
                ))

            ws_node_id = f"workspace_{self.dataset_name}_{recipe_name}"
            ls.add_node(LineageNode(
                id=ws_node_id,
                type="workspace",
                name=workspace_path,
            ))
            ls.add_edge(LineageEdge(
                source_id=recipe_node_id,
                target_id=ws_node_id,
                relation="builds",
            ))
        except Exception as e:
            print(f"  [Lineage Warn] Failed to write lineage: {e}")


# ──────────────────────────────────────────────
#  Convenience functions
# ──────────────────────────────────────────────


def build_workspace(
    dataset_name: str,
    recipe_name: str = "default",
    profile: Optional[BuildProfile] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> WorkspaceBuildResult:
    """One-shot build: load Recipe + Profile → build workspace."""
    builder = WorkspaceBuilder(dataset_name=dataset_name)
    return builder.build(
        recipe_name=recipe_name,
        profile=profile,
        output_dir=output_dir,
        force=force,
    )


def build_workspace_from_yaml(
    recipe_yaml: Path,
    profile_yaml: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dataset_name: str = "custom",
    force: bool = False,
) -> WorkspaceBuildResult:
    """Build workspace from raw YAML files (no registry needed)."""
    recipe = DatasetRecipe.from_yaml(recipe_yaml)
    profile = None
    if profile_yaml and profile_yaml.exists():
        profile = BuildProfile.load(profile_yaml)
    if profile is None:
        profile = BuildProfile.detection_default()

    builder = WorkspaceBuilder(dataset_name=dataset_name)
    return builder.build(
        recipe_name=recipe_yaml.stem.replace("recipe_", ""),
        profile=profile,
        output_dir=output_dir,
        force=force,
    )
