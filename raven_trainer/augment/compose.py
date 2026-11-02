"""Data augmentation pipeline with industrial-specific transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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
        return [{"name": t.name} for t in self.transforms]

    @classmethod
    def from_config(cls, config: List[Dict[str, Any]]) -> "AugmentPipeline":
        """Build pipeline from config dict."""
        # TODO: lookup transforms by name and instantiate
        return cls()


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
        # TODO: implement copy-paste logic
        return image, labels


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
        # TODO: implement occlusion simulation
        return image, labels


class IndustrialAugment(AugmentPipeline):
    """Standard augmentation pipeline for industrial vision.

    Enforces team-level augmentation standards:
    - Required: HSV, Flip, Scale, CopyPaste, Occlusion, MotionBlur
    - Forbidden: 180° flip (people don't walk upside-down in factories)
    """

    @classmethod
    def default(cls) -> "IndustrialAugment":
        """Create the standard industrial augmentation pipeline."""
        pipeline = cls()
        # Required industrial augmentations
        pipeline.add(CopyPaste(p=0.5))
        pipeline.add(OcclusionAug(p=0.3))
        return pipeline
