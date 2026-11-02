"""Data augmentation pipeline with industrial-specific transforms.

.. warning::
   ``CopyPaste`` and ``OcclusionAug`` are API placeholders: their pixel/label
   logic is not implemented. They raise ``NotImplementedError`` when called
   instead of silently returning the input unchanged, which previously made an
   "industrial" pipeline look like it was augmenting data when it was not.
   Training uses the backend's own augmentation (Ultralytics/MMDetection
   ``augment`` section), so nothing in the product path depends on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AugmentTransform(ABC):
    """Base class for a single augmentation transform."""

    name: str = "base"

    @abstractmethod
    def __call__(self, image: Any, labels: Any, **kwargs) -> Any:
        """Apply transform to image and labels."""
        ...


class AugmentPipeline:
    """Composable augmentation pipeline.

    Transforms are applied in order. The pipeline can be configured
    via YAML to enforce team-wide augmentation standards.
    """

    def __init__(self, transforms: Optional[List[AugmentTransform]] = None):
        self.transforms = transforms or []

    def add(self, transform: AugmentTransform) -> "AugmentPipeline":
        """Add a transform to the pipeline."""
        self.transforms.append(transform)
        return self

    def __call__(self, image: Any, labels: Any, **kwargs) -> Any:
        """Run all transforms sequentially."""
        for t in self.transforms:
            image, labels = t(image, labels, **kwargs)
        return image, labels

    def to_config(self) -> List[Dict[str, Any]]:
        """Serialize pipeline to config dict."""
        return [{"name": t.name, "params": dict(vars(t))} for t in self.transforms]

    @classmethod
    def from_config(cls, config: List[Dict[str, Any]]) -> "AugmentPipeline":
        """Build pipeline from config dict (inverse of :meth:`to_config`).

        Unknown transform names are rejected: silently dropping them made a
        configured pipeline run with fewer transforms than requested.
        """
        pipeline = cls()
        for entry in config:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid transform entry: {entry!r}")
            name = entry.get("name", "")
            factory = _TRANSFORM_REGISTRY.get(name)
            if factory is None:
                raise KeyError(
                    f"Unknown augmentation transform: {name!r}. "
                    f"Known transforms: {sorted(_TRANSFORM_REGISTRY)}"
                )
            params = entry.get("params") or {}
            pipeline.add(factory(**params))
        return pipeline


class CopyPaste(AugmentTransform):
    """Copy-Paste augmentation for instance segmentation / detection.

    Pastes object instances from one image onto another to increase
    instance diversity and handle occlusion.
    """

    name = "copy_paste"

    def __init__(self, p: float = 0.5, max_instances: int = 3):
        self.p = p
        self.max_instances = max_instances

    def __call__(self, image: Any, labels: Any, **kwargs) -> Any:
        raise NotImplementedError(
            "CopyPaste augmentation is not implemented yet; configure the "
            "backend's own copy-paste augmentation instead (e.g. Ultralytics "
            "`copy_paste`)."
        )


class OcclusionAug(AugmentTransform):
    """Simulates partial object occlusion for industrial scenarios.

    In factory environments, objects are frequently occluded by
    machinery, conveyor belts, or other objects. This transform
    adds synthetic occlusions to improve robustness.
    """

    name = "occlusion"

    def __init__(self, p: float = 0.3, max_occlusion_ratio: float = 0.4):
        self.p = p
        self.max_occlusion_ratio = max_occlusion_ratio

    def __call__(self, image: Any, labels: Any, **kwargs) -> Any:
        raise NotImplementedError(
            "OcclusionAug is not implemented yet; use the backend's own "
            "occlusion/mosaic augmentation instead."
        )


# Name -> factory used by AugmentPipeline.from_config()/_TRANSFORM_REGISTRY.
_TRANSFORM_REGISTRY: Dict[str, Any] = {
    "copy_paste": CopyPaste,
    "occlusion": OcclusionAug,
}


class IndustrialAugment(AugmentPipeline):
    """Standard augmentation pipeline for industrial vision.

    Enforces team-level augmentation standards:
    - Required: HSV, Flip, Scale, CopyPaste, Occlusion, MotionBlur
    - Forbidden: 180° flip (people don't walk upside-down in factories)
    """

    @classmethod
    def default(cls) -> "IndustrialAugment":
        """Create the standard industrial augmentation pipeline.

        Raises instead of returning a pipeline whose transforms are no-ops:
        the transforms below are not implemented, so a "default" pipeline would
        silently apply zero augmentation.
        """
        raise NotImplementedError(
            "IndustrialAugment.default() requires CopyPaste/OcclusionAug, which "
            "are not implemented; use the training backend's augmentation "
            "configuration (train.yaml `augment:`) instead."
        )
