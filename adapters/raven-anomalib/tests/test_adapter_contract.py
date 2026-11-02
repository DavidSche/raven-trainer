from __future__ import annotations

from pathlib import Path

from PIL import Image
from anomalib.models import Patchcore

from raven_anomalib import RavenAnomalibAdapter, _calibration_image_files, _leakage_check, _sample_hashes


def test_leakage_check_requires_sample_level_disjoint_sets(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "blue").save(second)
    hashes = {
        "fit": _sample_hashes([first]),
        "calibration": _sample_hashes([second]),
        "qualification": _sample_hashes([second]),
    }
    report = _leakage_check(hashes)
    assert report["status"] == "FAIL"
    assert report["pairs"]["calibration_qualification"]["intersection_count"] == 2


def test_calibration_image_collection_excludes_masks(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration"
    for split in ("normal", "anomaly", "masks"):
        split_dir = calibration / split
        split_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), "red").save(calibration / "normal" / "normal.png")
    Image.new("RGB", (2, 2), "blue").save(calibration / "anomaly" / "anomaly.png")
    Image.new("L", (2, 2), "white").save(calibration / "masks" / "anomaly.png")
    assert _calibration_image_files(calibration) == [
        calibration / "normal" / "normal.png",
        calibration / "anomaly" / "anomaly.png",
    ]


def test_patchcore_export_returns_checkpoint_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fit.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    adapter = RavenAnomalibAdapter({"model": "patchcore"})
    result = adapter.export(tmp_path / "patchcore", model_path=str(checkpoint))
    assert isinstance(result, dict)
    assert result["checkpoint_path"] == str(tmp_path / "patchcore.ckpt")
    assert result["status"] == "NOT_RMP"


def test_patchcore_checkpoint_is_lightning_loadable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split_name in (
        "train/normal",
        "calibration/normal",
        "calibration/anomaly",
        "qualification/normal",
    ):
        split = root / split_name
        split.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(root / "train/normal/first.png")
    Image.new("RGB", (8, 8), "blue").save(root / "calibration/normal/first.png")
    Image.new("RGB", (8, 8), "yellow").save(root / "calibration/anomaly/second.png")
    Image.new("RGB", (8, 8), "green").save(root / "qualification/normal/first.png")
    (tmp_path / "dataset.yaml").write_text(
        "task: anomaly\npath: dataset\ntrain: train\ncalibration: calibration\nqualification: qualification\n",
        encoding="utf-8",
    )
    adapter = RavenAnomalibAdapter(
        {
            "model": "patchcore",
            "backbone": "resnet18",
            "coreset_sampling_ratio": 1.0,
            "checkpoint_dir": str(tmp_path / "checkpoint"),
        },
    )
    result = adapter.train(tmp_path / "dataset.yaml")
    model = Patchcore.load_from_checkpoint(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert str(model.model.backbone) == "resnet18"
    import json

    fit_identity = json.loads((tmp_path / "checkpoint" / "evidence" / "fit_identity.json").read_text(encoding="utf-8"))
    leakage = json.loads((tmp_path / "checkpoint" / "evidence" / "data_leakage_check.json").read_text(encoding="utf-8"))
    assert fit_identity["memory_bank_hash"].startswith("sha256:")
    assert fit_identity["memory_bank_count"] == int(model.model.memory_bank.shape[0])
    assert fit_identity["sample_counts"]["calibration"] == 2
    assert leakage["sample_counts"]["calibration"] == 2
