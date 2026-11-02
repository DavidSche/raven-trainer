"""Augmentation pipeline (all transforms live in :mod:`raven_trainer.augment.compose`)."""

from .compose import AugmentPipeline, CopyPaste, OcclusionAug, IndustrialAugment

__all__ = ["AugmentPipeline", "CopyPaste", "OcclusionAug", "IndustrialAugment"]
