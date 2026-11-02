"""Unit tests for the shared CLI model locator (P0-3 regression guard).

Ensures train/export/eval all locate rf-detr ``.pth`` weights the same way
they locate YOLO ``.pt`` weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raven_trainer.cli._models import (  # noqa: E402
    latest_model_path,
    list_model_versions,
    next_model_path,
)


def test_list_model_versions_pt_and_pth(tmp_path):
    (tmp_path / "v1.0.pt").write_bytes(b"a")
    (tmp_path / "v1.0.pth").write_bytes(b"b")
    (tmp_path / "v2.0.pt").write_bytes(b"c")
    versions = list_model_versions(tmp_path)
    assert [p.name for p in versions] == ["v1.0.pt", "v1.0.pth", "v2.0.pt"]


def test_latest_model_path_pth(tmp_path):
    (tmp_path / "v1.0.pt").write_bytes(b"a")
    (tmp_path / "v1.0.pth").write_bytes(b"b")
    assert latest_model_path(tmp_path).name == "v1.0.pth"


def test_latest_model_path_none_when_empty(tmp_path):
    assert latest_model_path(tmp_path) is None


def test_next_model_path_increments_pt(tmp_path):
    (tmp_path / "v1.0.pt").write_bytes(b"a")
    assert next_model_path(tmp_path).name == "v1.1.pt"


def test_next_model_path_increments_pth(tmp_path):
    (tmp_path / "v1.0.pth").write_bytes(b"a")
    assert next_model_path(tmp_path).name == "v1.1.pth"


def test_next_model_path_fallback_v10(tmp_path):
    assert next_model_path(tmp_path).name == "v1.0.pt"


def test_next_model_path_with_version_keeps_ext(tmp_path):
    (tmp_path / "v1.0.pth").write_bytes(b"a")
    assert next_model_path(tmp_path, version="2.0").name == "v2.0.pth"
