"""Raven Trainer — Industrial Vision Dataset & Model Training Platform.

Core modules:
  data_asset       — Data Asset Management (Registry, Version, Metadata, Tag, Recipe, Doctor, Lineage)
  ingestion        — Data Ingestion Pipeline (import, sample, clean, dedup, scene, report)
  adapters         — pluggable model backends (raven-ultralytics, etc.)
  cli              — raven-train command-line interface

Dataset Pipeline (the core value):
  Raw Video → Ingest → Register → Version → Recipe → Workspace → Trainer → Export

Usage:
    raven-train dataset create person_detection
    raven-train dataset version person_detection --create
    raven-train dataset doctor person_detection
    raven-train dataset lineage person_detection
    raven-train ingest run data/raw
    raven-train new --name person_intrusion --task detection
    raven-train train project.yaml
    raven-train export project.yaml
"""

__version__ = "0.5.0"
