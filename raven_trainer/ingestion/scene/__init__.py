"""Layer 5: Scene Analysis -- cluster frames by visual similarity.

Extracts features using a lightweight model (ResNet-18 or similar),
runs KMeans clustering, and produces per-cluster statistics.

This enables balanced sampling across scenes and reveals
distribution gaps (e.g. all daytime, no night frames).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import ImportedSample, IngestionReport


class SceneAnalyzer:
    """Cluster frames by visual content and produce distribution stats."""

    def __init__(self, n_clusters: int = 8, feature_model: str = "resnet18"):
        self.n_clusters = n_clusters
        self.feature_model = feature_model

    def analyze(self, samples: List[ImportedSample]) -> List[ImportedSample]:
        """Run scene clustering.

        TODO: implement with torchvision + sklearn:
          1. Load ResNet-18 (without final FC layer) as feature extractor
          2. Extract feature vectors for each sample
          3. Run PCA to 64 dims (speed)
          4. Run KMeans(n_clusters) to assign scene labels
          5. Set sample.scene_label
        """
        raise NotImplementedError("Scene analysis requires torch + sklearn")

    def compute_distribution(self, samples: List[ImportedSample]) -> Dict[str, int]:
        """Count samples per scene label."""
        dist: Dict[str, int] = {}
        for s in samples:
            label = s.scene_label or "unknown"
            dist[label] = dist.get(label, 0) + 1
        return dist


def extract_features(image_path: Path) -> List[float]:
    """Extract feature vector from image using pretrained model.
    TODO: implement with torchvision ResNet
    """
    raise NotImplementedError("Requires torch + torchvision")
