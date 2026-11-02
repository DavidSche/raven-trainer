"""Raven Trainer — Anomalib (anomaly detection) backend adapter."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
import yaml

from raven_trainer.adapters.base import ModelAdapter, ModelInfo, ModelTask


class RavenAnomalibAdapter(ModelAdapter):
    """Adapter for Anomalib anomaly detection models."""

    backend_name = "anomalib"

    @classmethod
    def info(cls) -> ModelInfo:
        return ModelInfo(
            backend="anomalib",
            task=ModelTask.ANOMALY,
            framework="anomalib",
            supported_models=["patchcore", "fastflow", "efficientad", "padim"],
            description="Anomalib anomaly detection models",
        )

    def train(self, dataset_cfg: str | Path, **kwargs) -> Any:
        """Fit PatchCore and emit evidence; production RMP remains raven-export's responsibility."""
        from importlib.metadata import version as package_version
        from pathlib import Path as RuntimePath

        import lightning.pytorch as pl
        from anomalib.models import Patchcore

        config_path = Path(dataset_cfg)
        with config_path.open("r", encoding="utf-8") as handle:
            dataset = yaml.safe_load(handle) or {}
        if dataset.get("task") != "anomaly":
            raise ValueError("raven-anomalib PatchCore requires task: anomaly")
        if self.config.get("model") != "patchcore":
            raise ValueError("this anomalib adapter invocation only supports model: patchcore")
        root = (config_path.parent / dataset.get("path", "datasets")).resolve()
        train_root = root / dataset.get("train", "train")
        calibration_root = root / dataset.get("calibration", "calibration")
        qualification_root = root / dataset.get("qualification", "qualification")
        image_paths = {
            "fit": _image_files(train_root / "normal"),
            "calibration": _calibration_image_files(calibration_root),
            "qualification": _qualification_image_files(qualification_root),
        }
        if not image_paths["fit"]:
            raise ValueError(f"PatchCore fit set is empty: {image_paths['fit']}")
        if not image_paths["calibration"]:
            raise ValueError("PatchCore calibration set is empty")
        if not image_paths["qualification"]:
            raise ValueError(f"PatchCore qualification set is empty: {image_paths['qualification']}")
        seed = int(self.config.get("seed", kwargs.get("seed", 20260923)))
        backbone = str(self.config.get("backbone", "resnet18"))
        coreset_ratio = float(self.config.get("coreset_sampling_ratio", 0.1))
        pre_trained_config = self.config.get("pre_trained", True)
        pre_trained = pre_trained_config is not False and str(pre_trained_config).lower() != "false"
        if backbone not in ("resnet18", "wide_resnet50_2"):
            raise ValueError(f"unsupported PatchCore backbone: {backbone}")
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        _set_seed(seed)
        model = Patchcore(
            backbone=backbone,
            layers=("layer2", "layer3"),
            pre_trained=pre_trained,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=1,
            pre_processor=False,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        model.model.train()
        with torch.no_grad():
            for path in image_paths["fit"]:
                embedding = model.model(_load_patchcore_image(path, (256, 256)))
                model.model.embedding_store.append(embedding)
        model.fit()
        bank = model.model.memory_bank.detach().cpu().contiguous()
        if bank.ndim != 2 or bank.shape[0] == 0 or not torch.isfinite(bank).all():
            raise ValueError("PatchCore fit produced an empty or invalid memory bank")
        expected_dim = 384 if backbone == "resnet18" else 1536
        if int(bank.shape[1]) != expected_dim:
            raise ValueError(f"PatchCore embedding dimension mismatch: {bank.shape[1]} != {expected_dim}")
        model = model.cpu()
        model.model.memory_bank = bank.to(torch.float32).contiguous()
        checkpoint_dir = Path(self.config.get("checkpoint_dir", root / "patchcore-checkpoint"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"patchcore-{backbone.replace('_', '-')}.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": model.hparams,
                "epoch": 0,
                "global_step": 0,
                "pytorch-lightning_version": pl.__version__,
                "loops": {},
            },
            checkpoint_path,
        )
        hashes = {name: _sample_hashes(paths) for name, paths in image_paths.items()}
        leakage = _leakage_check(hashes)
        if leakage["status"] != "PASS":
            raise ValueError(f"PatchCore data leakage check failed: {leakage['findings']}")
        evidence_dir = checkpoint_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "fit_identity.json").write_text(json.dumps(_fit_identity(model, hashes, seed), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (evidence_dir / "data_leakage_check.json").write_text(json.dumps(leakage, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "PASS",
            "model": "patchcore",
            "checkpoint_path": str(checkpoint_path),
            "fit_identity_path": str(evidence_dir / "fit_identity.json"),
            "data_leakage_check_path": str(evidence_dir / "data_leakage_check.json"),
            "memory_bank_count": int(bank.shape[0]),
            "seed": seed,
            "toolchain": {name: package_version(name) for name in ("anomalib", "torch", "torchvision")},
        }

    def export(self, output_path: str | Path, **kwargs) -> dict[str, Any]:
        """Return the fitted checkpoint; do not create an unvalidated production RMP."""
        checkpoint = kwargs.get("model_path") or self.config.get("checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("PatchCore export requires a fitted anomalib checkpoint via model_path/checkpoint")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        destination = output_path.with_suffix(".ckpt")
        destination.write_bytes(Path(checkpoint).read_bytes())
        return {
            "checkpoint_path": str(destination),
            "status": "NOT_RMP",
            "message": "PatchCore checkpoint exported; use raven-export anomaly_export.export_rmp for production RMP",
        }


def _image_files(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _calibration_image_files(root: Path) -> list[Path]:
    """Collect calibration images only; masks are labels and are not samples."""
    return _image_files(root / "normal") + _image_files(root / "anomaly")


def _qualification_image_files(root: Path) -> list[Path]:
    """Collect held-out qualification images for leakage evidence."""
    normal = _image_files(root / "normal")
    anomaly_dir = root / "anomaly"
    return normal + _image_files(anomaly_dir) if anomaly_dir.is_dir() else normal


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _load_patchcore_image(path: Path, input_size: tuple[int, int]) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms import InterpolationMode, functional

    with Image.open(path) as image:
        source = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)
    resized = functional.resize(source, list(input_size), interpolation=InterpolationMode.BILINEAR, antialias=False)
    pixels = resized.numpy().astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy(((pixels - mean) / std)[None, ...])


def _sample_hashes(paths: list[Path]) -> dict[str, Any]:
    from PIL import Image, ImageOps

    file_hashes: list[str] = []
    decoded_hashes: list[str] = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hashes.append(f"sha256:{digest.hexdigest()}")
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image.convert("RGB"))
            decoded_hashes.append(f"sha256:{hashlib.sha256(rgb.tobytes()).hexdigest()}")
    return {"files": paths, "raw_file_sha256": file_hashes, "decoded_rgb_canonical_sha256": decoded_hashes}


def _leakage_check(hashes: dict[str, dict[str, Any]], *, checked_at: str | None = None) -> dict[str, Any]:
    from importlib.metadata import version as package_version

    raw = {name: set(value["raw_file_sha256"]) for name, value in hashes.items()}
    decoded = {name: set(value["decoded_rgb_canonical_sha256"]) for name, value in hashes.items()}
    pairs = (("fit", "calibration"), ("fit", "qualification"), ("calibration", "qualification"))
    pair_results = {}
    findings = []
    for left, right in pairs:
        key = f"{left}_{right}"
        counts = {"raw_file": len(raw[left] & raw[right]), "decoded_rgb": len(decoded[left] & decoded[right])}
        pair_results[key] = {"intersection_count": int(sum(counts.values())), **counts}
        if any(counts.values()):
            findings.append({"pair": key, **counts})
    return {
        "schema_version": "anomaly-data-leakage-check-v1",
        "status": "PASS" if not findings else "FAIL",
        "sample_counts": {name: len(value["files"]) for name, value in hashes.items()},
        "hash_set_digests": {name: f"sha256:{hashlib.sha256('|'.join(sorted(value)).encode()).hexdigest()}" for name, value in raw.items()},
        "pairs": pair_results,
        "findings": findings,
        "decoder": {"format": "PIL RGB", "exif_transpose": True, "layout": "HxWx3-row-major"},
        "toolchain": {name: package_version(name) for name in ("anomalib", "torch", "torchvision", "timm")},
        "checked_at": checked_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _fit_identity(model: Any, hashes: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    from importlib.metadata import version as package_version

    bank = model.model.memory_bank.detach().cpu()
    return {
        "schema_version": "patchcore-fit-identity-v1",
        "backbone": str(model.model.backbone),
        "layers": list(model.model.layers),
        "input_size": [256, 256],
        "sample_counts": {name: len(value["files"]) for name, value in hashes.items()},
        "dataset_fingerprint": f"sha256:{hashlib.sha256(json.dumps({name: value['decoded_rgb_canonical_sha256'] for name, value in hashes.items()}, sort_keys=True).encode()).hexdigest()}",
        "fit_sample_order_hash": f"sha256:{hashlib.sha256('|'.join(path.as_posix() for path in hashes['fit']['files']).encode()).hexdigest()}",
        "embedding_count": int(bank.shape[0]),
        "memory_bank_count": int(bank.shape[0]),
        "memory_bank_hash": f"sha256:{hashlib.sha256(bank.detach().numpy().tobytes()).hexdigest()}",
        "memory_bank_limit": {"max_validated": 65536, "max_validated_model_bytes": 134217728},
        "coreset_sampling_ratio": float(model.coreset_sampling_ratio),
        "seed": seed,
        "map_num_neighbors": 1,
        "toolchain": {name: package_version(name) for name in ("anomalib", "torch", "torchvision", "timm")},
    }
