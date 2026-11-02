"""Unit tests for raven_rfdetr.config — no GPU / no rfdetr install needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven_rfdetr.config import (  # noqa: E402
    MODEL_VARIANTS,
    _TRAIN_WHITELIST,
    ensure_rf_yolo_layout,
    filter_train_kwargs,
    labels_from_dataset,
    resolve_variant,
    validate_pretrain_weights,
)


# ---------------------------------------------------------------------------
# variant mapping
# ---------------------------------------------------------------------------

def test_resolve_variant_detection_resolutions():
    assert resolve_variant("rf-detr-nano").resolution == 384
    assert resolve_variant("rf-detr-small").resolution == 512
    assert resolve_variant("rf-detr-base").resolution == 560
    assert resolve_variant("rf-detr-medium").resolution == 576
    assert resolve_variant("rf-detr-large").resolution == 704


def test_resolve_variant_seg_resolutions():
    # P2-15: seg 分辨率与 detection 不同
    assert resolve_variant("rf-detr-seg-nano").resolution == 312
    assert resolve_variant("rf-detr-seg-small").resolution == 384
    assert resolve_variant("rf-detr-seg-medium").resolution == 432
    assert resolve_variant("rf-detr-seg-large").resolution == 504


def test_resolve_variant_alias_normalization():
    # rfdetr-* 别名归一化到 rf-detr-* canonical
    v = resolve_variant("rfdetr-medium")
    assert v.model_name == "rf-detr-medium"
    assert v.variant_cls == "RFDETRMedium"


def test_resolve_variant_deprecated_flag():
    assert resolve_variant("rf-detr-base").deprecated is True
    assert resolve_variant("rf-detr-medium").deprecated is False


def test_resolve_variant_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported rfdetr model"):
        resolve_variant("yolo26s")
    with pytest.raises(ValueError, match="model name is required"):
        resolve_variant(None)


def test_model_variants_registry_complete():
    for name in (
        "rf-detr-nano", "rf-detr-small", "rf-detr-base", "rf-detr-medium",
        "rf-detr-large", "rf-detr-seg-nano", "rf-detr-seg-small",
        "rf-detr-seg-medium", "rf-detr-seg-large",
    ):
        assert name in MODEL_VARIANTS


# ---------------------------------------------------------------------------
# data.yaml idempotent projection
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path: Path, names=None, include_test=True, path=None):
    data = {
        "path": path or str(tmp_path),
        "train": "train/images",
        "val": "val/images",
    }
    if include_test:
        data["test"] = ""  # Raven 常见 test: ''
    data["names"] = names if names is not None else {0: "person"}
    data["nc"] = len(data["names"])
    ds = tmp_path / "dataset.yaml"
    import yaml
    ds.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    return ds


def test_data_yaml_generated_and_idempotent(tmp_path):
    ds = _write_dataset(tmp_path)
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)

    r1 = ensure_rf_yolo_layout(tmp_path, ds)
    out = tmp_path / "data.yaml"
    assert out.exists()
    content1 = out.read_text(encoding="utf-8")

    # 幂等：内容一致不重写
    r2 = ensure_rf_yolo_layout(tmp_path, ds)
    assert out.read_text(encoding="utf-8") == content1
    assert r1 == r2


def test_data_yaml_strips_test_field(tmp_path):
    ds = _write_dataset(tmp_path, include_test=True)
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)

    ensure_rf_yolo_layout(tmp_path, ds)
    import yaml
    with open(tmp_path / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "test" not in data  # P1-10: 必须剔除 test


def test_data_yaml_names_list_normalized(tmp_path):
    # P2-21: 历史产物 names 可能是 list
    ds = _write_dataset(tmp_path, names=["person", "helmet"])
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)

    projected = ensure_rf_yolo_layout(tmp_path, ds)
    assert projected["names"] == {0: "person", 1: "helmet"}
    assert projected["nc"] == 2


def test_data_yaml_posix_path_on_windows(tmp_path):
    # P1-10: Windows 反斜杠 YAML 转义问题 — 必须是 posix 路径
    ds = _write_dataset(tmp_path)
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)

    projected = ensure_rf_yolo_layout(tmp_path, ds)
    # Windows 绝对路径的 posix 表示是 C:/...（正斜杠）；绝不允许反斜杠
    assert "\\" not in projected["path"]
    assert projected["path"].replace("/", "").isalnum() or ":" in projected["path"]
    assert projected["path"].endswith(str(tmp_path.name))


def test_data_yaml_overwrites_on_change(tmp_path):
    ds = _write_dataset(tmp_path, names={0: "person"})
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    ensure_rf_yolo_layout(tmp_path, ds)

    # 数据集换版本：names 变化必须覆盖并告警
    ds2 = _write_dataset(tmp_path, names={0: "person", 1: "car"})
    logs = []
    projected = ensure_rf_yolo_layout(tmp_path, ds2, log_func=logs.append)
    assert projected["nc"] == 2
    assert any("CHANGED" in line for line in logs)


def test_data_root_resolves_relative_to_cwd(tmp_path, monkeypatch):
    """Raven project dataset.yaml 的 path 常是相对 raven-trainer 根的逻辑路径
    （如 data/workspace/xxx），需按 cwd 解析。"""
    import yaml

    # 构造：project 目录下 dataset.yaml，path 指向 cwd 下的 workspace
    proj = tmp_path / "proj"
    ws = tmp_path / "ws"
    (ws / "train" / "images").mkdir(parents=True)
    (ws / "val" / "images").mkdir(parents=True)
    proj.mkdir()
    ds = proj / "dataset.yaml"
    ds.write_text(yaml.dump({
        "path": "ws",  # 相对 cwd
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
        "nc": 1,
    }, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    projected = ensure_rf_yolo_layout(proj, ds)
    assert Path(projected["path"]).resolve() == ws.resolve()


def test_missing_dataset_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_rf_yolo_layout(tmp_path, tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# pretrain_weights validation (P1-8)
# ---------------------------------------------------------------------------

def test_pretrain_weights_local_path(tmp_path):
    w = tmp_path / "my_weights.pth"
    w.write_bytes(b"x")
    assert validate_pretrain_weights(str(w)) == str(w)


def test_pretrain_weights_official_name_ok():
    assert validate_pretrain_weights("rf-detr-medium.pth") == "rf-detr-medium.pth"


def test_pretrain_weights_url_rejected():
    with pytest.raises(ValueError, match="does not support URLs"):
        validate_pretrain_weights("https://example.com/w.pth")


def test_pretrain_weights_unknown_name_rejected():
    with pytest.raises(ValueError, match="not a local file and not an official"):
        validate_pretrain_weights("my-custom-weight.pth")


def test_pretrain_weights_none_ok():
    assert validate_pretrain_weights(None) is None


# ---------------------------------------------------------------------------
# whitelist filtering (TrainConfig extra="forbid")
# ---------------------------------------------------------------------------

def test_filter_train_kwargs_drops_unknown():
    cfg = {"epochs": 100, "lr": 1e-4, "typ0_key": 1, "model": "rf-detr-medium"}
    logs = []
    out = filter_train_kwargs(cfg, log_func=logs.append)
    assert "epochs" in out
    assert "lr" in out
    assert "typ0_key" not in out
    assert "model" not in out  # 适配器消费键不转发
    assert any("dropping unknown" in line for line in logs)


def test_train_whitelist_matches_trainconfig_fields():
    """Every whitelisted key must be a real rfdetr TrainConfig field.

    TrainConfig is extra="forbid": any non-field key forwarded to model.train()
    raises ValidationError and kills training. The whitelist is the adapter's
    last line of defense against wrong fills — it must be regenerated whenever
    upstream adds/renames TrainConfig fields (e.g. val_interval -> eval_interval).
    """
    try:
        from rfdetr.config import TrainConfig
    except ImportError:
        pytest.skip("rfdetr not installed — cannot verify against TrainConfig")
    fields = set(TrainConfig.model_fields.keys())
    invalid = sorted(k for k in _TRAIN_WHITELIST if k not in fields)
    assert invalid == [], (
        f"_TRAIN_WHITELIST contains keys that are not rfdetr TrainConfig fields "
        f"(would crash TrainConfig extra='forbid'): {invalid}"
    )


def test_filter_train_kwargs_keeps_known():
    cfg = {
        "epochs": 50, "batch_size": 4, "num_workers": 2, "lr": 1e-4,
        "weight_decay": 1e-4, "use_ema": True, "progress_bar": "tqdm",
        "early_stopping": True, "early_stopping_patience": 10,
        "resolution": 576, "device": "cpu",
    }
    out = filter_train_kwargs(cfg)
    for k in ("epochs", "batch_size", "num_workers", "lr", "weight_decay",
              "use_ema", "progress_bar", "early_stopping", "early_stopping_patience"):
        assert k in out
    # resolution / device 由适配器单独消费，不转发
    assert "resolution" not in out
    assert "device" not in out


# ---------------------------------------------------------------------------
# labels extraction
# ---------------------------------------------------------------------------

def test_labels_from_dataset(tmp_path):
    ds = _write_dataset(tmp_path, names={0: "person", 1: "helmet"})
    assert labels_from_dataset(ds) == ["person", "helmet"]
