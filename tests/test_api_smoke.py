"""API smoke tests: real FastAPI app against a throwaway data root.

These pin the fixes that are easy to regress:
  * ``InvalidNameError`` → HTTP 400 (instead of 500 / silent traversal).
  * ``/datasets/{name}/lineage`` returns nodes+edges (previously crashed with
    ``TypeError`` because ``get_lineage`` was called without a node id).
  * Duplicate version tag → 409, not an overwrite.
  * ``/api/health`` reachable and UI catch-all does not shadow the API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from raven_trainer.api.server import app  # noqa: E402
from raven_trainer.config import R  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "root"
    data = root / "data"
    for name in (
        "storage",
        "storage/images",
        "storage/videos",
        "datasets",
        "projects",
        "recipes",
        "workspace",
        "ingestion",
        "ingestion/runs",
        "ingestion/profiles",
        "cache",
        "pipeline/profiles",
    ):
        (data / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(R, "DATA", data, raising=False)
    monkeypatch.setattr(R, "PROJECT_ROOT", root, raising=False)
    monkeypatch.setattr(R, "STORAGE", data / "storage", raising=False)
    monkeypatch.setattr(R, "STORAGE_IMAGES", data / "storage" / "images", raising=False)
    monkeypatch.setattr(R, "STORAGE_VIDEOS", data / "storage" / "videos", raising=False)
    monkeypatch.setattr(R, "DATASETS", data / "datasets", raising=False)
    monkeypatch.setattr(R, "PROJECTS", data / "projects", raising=False)
    monkeypatch.setattr(R, "RECIPES", data / "recipes", raising=False)
    monkeypatch.setattr(R, "WORKSPACE", data / "workspace", raising=False)
    monkeypatch.setattr(R, "INGESTION", data / "ingestion", raising=False)
    monkeypatch.setattr(R, "INGESTION_RUNS", data / "ingestion" / "runs", raising=False)
    monkeypatch.setattr(R, "INGESTION_PROFILES", data / "ingestion" / "profiles", raising=False)
    monkeypatch.setattr(R, "CACHE", data / "cache", raising=False)
    monkeypatch.setattr(R, "PIPELINE_PROFILES", data / "pipeline" / "profiles", raising=False)
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_dataset_and_list(client):
    created = client.post("/api/v1/datasets", json={"name": "demo_ds"})
    assert created.status_code == 200, created.text
    listed = client.get("/api/v1/datasets")
    assert [d["name"] for d in listed.json()] == ["demo_ds"]


@pytest.mark.parametrize("bad_name", ["../escape", "a/b", "..", "   "])
def test_invalid_dataset_name_returns_400(client, bad_name):
    # Path params: httpx encodes '/' so '..' style traversal is the real case.
    resp = client.get(f"/api/v1/datasets/{bad_name}/lineage")
    assert resp.status_code in (400, 404), resp.text


def test_lineage_endpoint_returns_graph(client):
    client.post("/api/v1/datasets", json={"name": "lineage_ds"})
    resp = client.get("/api/v1/datasets/lineage_ds/lineage")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"nodes": [], "edges": []}


def test_duplicate_version_returns_409(client):
    client.post("/api/v1/datasets", json={"name": "ver_ds"})
    first = client.post("/api/v1/datasets/ver_ds/versions", json={"version": "1.0.0"})
    assert first.status_code == 200, first.text
    second = client.post("/api/v1/datasets/ver_ds/versions", json={"version": "1.0.0"})
    assert second.status_code == 409, second.text


def test_delete_missing_dataset_returns_404(client):
    assert client.delete("/api/v1/datasets/nope").status_code == 404


def test_unknown_job_returns_404(client):
    assert client.get("/api/v1/jobs/../../etc/passwd").status_code in (404, 400)
    assert client.get("/api/v1/jobs/train_9999").status_code == 404


def test_no_blocking_work_on_the_event_loop():
    """Every HTTP route must be a sync endpoint.

    The handlers do blocking file/CPU work (workspace builds, YOLO prelabel,
    ffprobe, dataset analysis). Declaring them ``async def`` without awaiting
    anything parks all that work on the single event loop thread and stalls
    every other request, including health checks. Sync endpoints are executed
    in Starlette's threadpool instead.
    """
    import inspect

    from fastapi.routing import APIRoute

    offenders = [
        f"{sorted(route.methods)[0]} {route.path} -> {route.endpoint.__name__}"
        for route in app.routes
        if isinstance(route, APIRoute) and inspect.iscoroutinefunction(route.endpoint)
    ]
    assert offenders == []


# ── Round 3: annotation/feedback robustness ──────────────────────────────────


def _make_version_with_images(client, name: str, count: int = 2):
    """Create a dataset version whose manifest points at real files."""
    import json

    client.post("/api/v1/datasets", json={"name": name})
    client.post(f"/api/v1/datasets/{name}/versions", json={"version": "v1"})
    version_dir = R.dataset_version_dir(name, "v1")
    records = []
    for i in range(count):
        rel = f"data/storage/images/cam/f{i}.jpg"
        img = R.PROJECT_ROOT / rel
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8\xff\xd9")
        records.append({"storage_path": rel, "image_id": f"img_{i}", "filename": f"f{i}.jpg"})
    with open(version_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return version_dir


def test_browse_tolerates_unresolvable_manifest_records(client):
    """One bad manifest path must not 500 the whole listing."""
    version_dir = _make_version_with_images(client, "browse_ds", 2)
    with open(version_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write('{"storage_path": "../../etc/passwd"}\n')
        f.write("not-json\n")
    resp = client.get("/api/v1/datasets/browse_ds/versions/v1/browse")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 3
    assert body["missing_files"] == 1
    assert body["images"][2]["available"] is False


def test_serve_image_returns_404_for_escaping_record(client):
    version_dir = _make_version_with_images(client, "serve_ds", 1)
    with open(version_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write('{"storage_path": "../../etc/passwd"}\n')
    resp = client.get("/api/v1/datasets/serve_ds/versions/v1/image/0")
    assert resp.status_code == 404, resp.text


def test_serve_image_uses_png_media_type(client):
    version_dir = _make_version_with_images(client, "png_ds", 1)
    import json

    rel = "data/storage/images/cam/f0.png"
    (R.PROJECT_ROOT / rel).write_bytes(b"\x89PNG\r\n\x1a\n")
    with open(version_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"storage_path": rel, "image_id": "img_0"}) + "\n")
    resp = client.get("/api/v1/datasets/png_ds/versions/v1/image/0")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/png")


def test_update_annotations_rejects_negative_class_id(client):
    _make_version_with_images(client, "ann_ds", 1)
    resp = client.put(
        "/api/v1/datasets/ann_ds/versions/v1/annotations/0",
        json={"boxes": [[-1, 0.5, 0.5, 0.2, 0.2]]},
    )
    assert resp.status_code == 400, resp.text


def test_batch_annotation_rejects_unknown_operation(client):
    """An unknown operation used to return 200 when no image had labels."""
    _make_version_with_images(client, "batch_ds", 2)
    resp = client.post(
        "/api/v1/datasets/batch_ds/versions/v1/batch-annotations",
        json={"operation": "explode"},
    )
    assert resp.status_code == 400, resp.text


def test_feedback_approve_missing_item_returns_404(client):
    client.post("/api/v1/datasets", json={"name": "fb_ds"})
    resp = client.post(
        "/api/v1/feedback/fb_ds/approve",
        json={"category": "false_positive", "filename": "ghost.jpg"},
    )
    assert resp.status_code == 404, resp.text


def test_prelabel_rejects_model_path_outside_data_roots(client, tmp_path):
    _make_version_with_images(client, "pre_ds", 1)
    outside = tmp_path / "evil.pt"
    outside.write_bytes(b"x")
    resp = client.post(
        "/api/v1/datasets/pre_ds/versions/v1/prelabel",
        json={"model_path": str(outside)},
    )
    assert resp.status_code == 400, resp.text


def test_report_id_with_wildcards_is_rejected(client):
    resp = client.get("/api/v1/reports/*secret*")
    assert resp.status_code in (400, 404), resp.text
