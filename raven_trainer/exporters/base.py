"""Base exporter for model deployment formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExportResult:
    """Result metadata for a model export."""

    output_path: Path
    format: str
    model_name: str
    model_version: str
    sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[List[int]] = None
    output_names: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


class BaseExporter(ABC):
    """Abstract base for all model exporters.

    Every export includes mandatory metadata:
    - Model name & version
    - Label list
    - SHA256 checksum
    - Input / output shapes
    """

    def __init__(self, model: Any, config: Dict[str, Any]):
        self.model = model
        self.config = config

    @abstractmethod
    def export(self, output_path: str | Path, **kwargs) -> ExportResult:
        """Export model to the target format."""
        ...

    def verify(self, result: ExportResult) -> bool:
        """Verify exported model integrity."""
        # TODO: implement SHA256 check and shape validation
        return True
