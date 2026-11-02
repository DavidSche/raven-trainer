"""Regression tests for the path-safety / data-integrity hardening pass.

Covers:
  * ``validate_name`` — the single gate for user-supplied path components.
  * ``R.resolve_storage_path`` — manifests must not become arbitrary file reads.
  * ``VersionManager`` — append-only versions, collision-free numbering.
  * ``JobManager`` — traversal-safe job ids, corrupt-file tolerance.
  * ``TaskManager`` — traversal-safe annotation task ids.
  * ``CvatImporter`` — CVAT label lookup actually maps names to class ids.
  * ``LineageStore`` — graph dump used by the API.
  * ``ExperimentManager`` — traversal-safe experiment ids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raven_trainer.config import InvalidNameError, R, validate_name  # noqa: E402
from raven_trainer.data_asset.job import JobManager, JobStatus, JobType  # noqa: E402
from raven_trainer.data_asset.lineage import (  # noqa: E402
    LineageEdge,
    LineageNode,
    LineageStore,
)
from raven_trainer.data_asset.version import VersionManager  # noqa: E402
from raven_trainer.datasets.converters.cvat import CvatImporter  # noqa: E402


# ── validate_name ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["dataset_a", "v1", "v1.0.0", "my-dataset", "abc123"])
def test_validate_name_accepts_safe_names(name):
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        " padded",
        "padded ",
        ".",
        "..",
        "../escape",
        "..\\escape",
        "a/b",
        "a\\b",
        "/absolute",
        "C:evil",
        ".hidden",
        "tab\there",
        "nul\x00",
        "x" * 129,
    ],
)
def test_validate_name_rejects_unsafe_names(name):
    with pytest.raises(InvalidNameError):
        validate_name(name)


def test_invalid_name_is_value_error():
    # FastAPI's handler keys on ValueError descendants; keep that contract.
    assert issubclass(InvalidNameError, ValueError)


# ── resolve_storage_path ─────────────────────────────────────────────────────


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    """Point R.DATA (and friends) at a temporary tree."""
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(R, "DATA", data, raising=False)
    monkeypatch.setattr(R, "PROJECT_ROOT", root, raising=False)
    return data


def test_resolve_storage_path_allows_inside_data(data_root):
    target = data_root / "storage" / "images" / "cam" / "a.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert R.resolve_storage_path(str(target)) == target.resolve()


def test_resolve_storage_path_rejects_outside_data(data_root, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        R.resolve_storage_path(str(secret))


def test_resolve_storage_path_rejects_parent_segments(data_root):
    with pytest.raises(FileNotFoundError):
        R.resolve_storage_path("../../etc/passwd")


# ── VersionManager ───────────────────────────────────────────────────────────


def test_create_version_rejects_duplicate_tag(tmp_path):
    vm = VersionManager(tmp_path / "asset")
    vm.create_version(version_tag="v1")
    with pytest.raises(FileExistsError):
        vm.create_version(version_tag="v1")


def test_next_version_str_uses_max_plus_one(tmp_path):
    vm = VersionManager(tmp_path / "asset")
    vm.create_version(version_tag="v1")
    vm.create_version(version_tag="v2")
    vm.create_version(version_tag="v3")
    # Simulate a deletion of a middle version: len(existing)+1 would collide.
    import shutil

    shutil.rmtree(vm.get_version_dir("v2"))
    assert vm.next_version_str() == "v4"


def test_create_version_rejects_unsafe_tag(tmp_path):
    vm = VersionManager(tmp_path / "asset")
    with pytest.raises(InvalidNameError):
        vm.create_version(version_tag="../escape")


# ── JobManager ───────────────────────────────────────────────────────────────


@pytest.fixture()
def job_manager(data_root):
    mgr = JobManager()
    mgr.jobs_dir = data_root / "jobs"
    mgr.jobs_dir.mkdir(parents=True, exist_ok=True)
    return mgr


def test_job_manager_rejects_traversal_id(job_manager):
    # Invalid ids are treated as "not found" rather than raised: the API maps
    # None to 404 and never touches the filesystem with attacker input.
    assert job_manager.get("../../etc/passwd") is None
    assert job_manager.get("train_0001/../../x") is None


def test_job_manager_skips_corrupt_and_sidecar_files(job_manager):
    good = job_manager.create(JobType.TRAIN, {"epochs": 1})
    (job_manager.jobs_dir / f"{good.job_id}_result.json").write_text("{}", encoding="utf-8")
    (job_manager.jobs_dir / "train_9999.json").write_text("{not json", encoding="utf-8")
    jobs = job_manager.list_jobs()
    assert [j.job_id for j in jobs] == [good.job_id]
    assert jobs[0].status is JobStatus.PENDING


# ── TaskManager ──────────────────────────────────────────────────────────────


def test_task_manager_rejects_traversal_id(data_root, monkeypatch):
    datasets = data_root / "datasets"
    datasets.mkdir()
    monkeypatch.setattr(R, "DATASETS", datasets, raising=False)
    from raven_trainer.data_asset.annotation.task import TaskManager

    tm = TaskManager("dummy", "v1")
    with pytest.raises(InvalidNameError):
        tm.get_task("../../escape")


# ── CvatImporter regression ──────────────────────────────────────────────────


def test_cvat_importer_maps_label_names_to_class_ids(tmp_path):
    """Regression: the importer used a reversed label map, dropping every box."""
    cvat_xml = tmp_path / "annotations.xml"
    cvat_xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="frame.jpg" width="100" height="100">
    <box label="person" xtl="10" ytl="10" xbr="50" ybr="50"/>
    <box label="car" xtl="60" ytl="60" xbr="90" ybr="90"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"x")
    out_dir = tmp_path / "labels"

    processed = CvatImporter({"person": 0, "car": 1}).cvat_xml_to_yolo(
        cvat_xml, image_dir, out_dir
    )

    assert processed == [image_dir / "frame.jpg"]
    lines = (out_dir / "frame.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in lines] == ["0", "1"]


# ── LineageStore ─────────────────────────────────────────────────────────────


def test_lineage_store_get_graph_returns_all_nodes(tmp_path):
    store = LineageStore(tmp_path)
    store.add_node(LineageNode(id="video_1", type="video", name="cam1"))
    store.add_node(LineageNode(id="dataset_1", type="dataset", name="ds"))
    store.add_edge(LineageEdge(source_id="video_1", target_id="dataset_1", relation="derived_from"))

    graph = store.get_graph()
    assert {n.id for n in graph["nodes"]} == {"video_1", "dataset_1"}
    assert len(graph["edges"]) == 1
    assert store.get_lineage("dataset_1")["upstream"][0]["relation"] == "derived_from"


def test_lineage_store_roundtrip(tmp_path):
    store = LineageStore(tmp_path)
    store.add_node(LineageNode(id="a", type="video", name="a"))
    reloaded = LineageStore(tmp_path)
    assert [n.id for n in reloaded.get_graph()["nodes"]] == ["a"]


# ── ExperimentManager ────────────────────────────────────────────────────────


def test_experiment_manager_rejects_traversal_id(data_root, monkeypatch):
    projects = data_root / "projects"
    projects.mkdir()
    monkeypatch.setattr(R, "PROJECTS", projects, raising=False)
    from raven_trainer.experiments import ExperimentManager

    mgr = ExperimentManager("proj")
    with pytest.raises(InvalidNameError):
        mgr.get("../../escape")


def test_experiment_manager_creates_sequential_ids(data_root, monkeypatch):
    projects = data_root / "projects"
    projects.mkdir()
    monkeypatch.setattr(R, "PROJECTS", projects, raising=False)
    from raven_trainer.experiments import ExperimentManager

    mgr = ExperimentManager("proj")
    first = mgr.create(model="yolo26n", backend="ultralytics")
    second = mgr.create(model="yolo26n", backend="ultralytics")
    assert (first.experiment_id, second.experiment_id) == ("exp_0001", "exp_0002")
    assert mgr.delete(first.experiment_id) is True
    assert mgr.delete(first.experiment_id) is False
    # After deleting exp_0001 a new run must not reuse the id.
    third = mgr.create(model="yolo26n", backend="ultralytics")
    assert third.experiment_id == "exp_0003"


# ── Ingestion profile tolerance ─────────────────────────────────────────────


def test_profile_filter_kwargs_drops_unknown_keys():
    from raven_trainer.ingestion.profile import _filter_kwargs, SourceRef

    filtered = _filter_kwargs(SourceRef, {"name": "cam1", "path": "images/cam1", "bogus": 1})
    assert filtered == {"name": "cam1", "path": "images/cam1"}
    assert all(isinstance(k, str) for k in filtered)


def test_profile_yaml_with_unknown_keys_does_not_crash(tmp_path):
    import yaml

    from raven_trainer.ingestion.profile import IngestionProfile

    payload = {
        "name": "p1",
        "sources": [{"name": "cam1", "path": "images/cam1", "unexpected": "x"}],
        "quality": {"blur_laplacian_threshold": 10, "mystery": True},
        "dedup": {"method": "dhash", "unknown": 1},
        "no_such_field": 42,
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    profile = IngestionProfile.from_yaml(path)
    assert profile.name == "p1"
    assert profile.sources[0].name == "cam1"


def test_job_result_sidecar_is_not_listed_as_job(job_manager):
    job = job_manager.create(JobType.INGEST, {})
    (job_manager.jobs_dir / f"{job.job_id}_result.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8"
    )
    assert [j.job_id for j in job_manager.list_jobs()] == [job.job_id]


# ── Round 3: version ordering, atomicity, parent lineage ──────────────────────


def test_list_versions_orders_numerically(tmp_path):
    """v10 must sort after v9 — string sorting silently broke "latest"."""
    vm = VersionManager(tmp_path / "asset")
    for n in (1, 2, 9, 10, 11):
        vm.create_version(version_tag=f"v{n}")
    assert [v.version for v in vm.list_versions()] == ["v11", "v10", "v9", "v2", "v1"]
    assert vm.next_version_str() == "v12"


def test_explicit_version_tag_keeps_parent_lineage(tmp_path):
    vm = VersionManager(tmp_path / "asset")
    vm.create_version(version_tag="v1")
    second = vm.create_version(version_tag="release-1")
    assert second.parent_version == "v1"


def test_failed_version_creation_leaves_no_partial_dir(tmp_path):
    """A failure mid-write must not leave a half-built version directory."""
    vm = VersionManager(tmp_path / "asset")

    class Boom:
        def __iter__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        vm.create_version(manifest_records=Boom())
    assert vm.list_versions() == []
    assert not [d for d in vm.versions_dir.iterdir() if d.is_dir()]


# ── Round 3: annotation / task integrity ─────────────────────────────────────


def test_task_manager_refuses_to_overwrite_existing_task(data_root, monkeypatch):
    from raven_trainer.data_asset.annotation.task import AnnotationTask, TaskManager

    monkeypatch.setattr(R, "DATASETS", data_root / "datasets", raising=False)
    tm = TaskManager("ds", "v1")
    tm.create_task(AnnotationTask(task_id="t1", platform="cvat", name="first"))
    with pytest.raises(FileExistsError):
        tm.create_task(AnnotationTask(task_id="t1", platform="cvat", name="second"))


def test_task_load_tolerates_unknown_and_missing_keys(data_root, monkeypatch):
    from raven_trainer.data_asset.annotation.task import AnnotationTask, TaskManager

    monkeypatch.setattr(R, "DATASETS", data_root / "datasets", raising=False)
    tm = TaskManager("ds", "v1")
    path = tm.tasks_dir / "task_extra.yaml"
    path.write_text(
        "task_id: extra\nplatform: cvat\nname: n\nfuture_field: 1\n", encoding="utf-8"
    )
    task = AnnotationTask.load(path)
    assert task.task_id == "extra"
    # An empty file must fail with a clear ValueError, not a TypeError.
    empty = tm.tasks_dir / "task_empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        AnnotationTask.load(empty)


# ── Round 3: metadata store ──────────────────────────────────────────────────


def test_metadata_query_rejects_unknown_column(tmp_path):
    from raven_trainer.data_asset.metadata import MetadataStore

    with MetadataStore(tmp_path / "md") as store:
        with pytest.raises(ValueError):
            store.query(**{"camera = 'x' OR 1": "y"})


def test_metadata_store_after_close_raises_runtime_error(tmp_path):
    from raven_trainer.data_asset.metadata import MetadataStore

    store = MetadataStore(tmp_path / "md")
    store.close()
    with pytest.raises(RuntimeError):
        store.count()


# ── Round 3: augmentation placeholders must not silently no-op ───────────────


def test_unknown_augment_transform_is_rejected():
    from raven_trainer.augment import AugmentPipeline

    with pytest.raises(KeyError):
        AugmentPipeline.from_config([{"name": "nonexistent"}])


def test_augment_pipeline_config_roundtrip():
    from raven_trainer.augment import AugmentPipeline, CopyPaste

    pipeline = AugmentPipeline([CopyPaste(p=0.7)])
    restored = AugmentPipeline.from_config(pipeline.to_config())
    assert [t.name for t in restored.transforms] == ["copy_paste"]
    assert restored.transforms[0].p == 0.7


def test_unimplemented_augment_transforms_raise():
    from raven_trainer.augment import CopyPaste, OcclusionAug

    for transform in (CopyPaste(), OcclusionAug()):
        with pytest.raises(NotImplementedError):
            transform("image", [])
