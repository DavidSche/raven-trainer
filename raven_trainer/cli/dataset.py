"""raven-train dataset — 数据资产管理 CLI。

子命令：
  dataset list             列出所有数据集
  dataset create <name>    创建新数据集
  dataset show <name>      查看数据集详情
  dataset version <name>   管理版本
  dataset tag <name>       管理标签
  dataset stats <name>     查看统计
  dataset recipe <name>    管理 Recipe
  dataset doctor <name>    健康检查
  dataset lineage <name>   查看血缘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import click

from ..config import R
from ..data_asset import (
    AssetStore, DatasetRegistry, AnnotationTask,
    VersionManager, MetadataStore,
    TagManager, DatasetRecipe, RecipeManager,
    DatasetDoctor, LineageStore,
)


def _get_registry() -> DatasetRegistry:
    store = AssetStore(R.DATASETS)
    return DatasetRegistry(store)


def _get_asset_dir(name: str) -> Path:
    return R.DATASETS / name


# -------------------------------------------------------------------
# dataset list
# -------------------------------------------------------------------

@click.group(name="dataset")
def cmd_dataset():
    """数据资产管理（Data Asset Management）。"""


@cmd_dataset.command(name="list")
def dataset_list():
    """列出所有已注册的数据集。"""
    registry = _get_registry()
    assets = registry.list()
    if not assets:
        click.echo("[Dataset] No datasets registered. Use 'raven-train dataset create <name>' to create one.")
        return

    click.echo(f"{'NAME':25s} {'TASK':15s} {'VERSION':10s} {'IMAGES':>8s} {'STATUS':12s}")
    click.echo("-" * 70)
    for a in assets:
        click.echo(f"{a.name:25s} {a.task:15s} {a.current_version:10s} {a.total_images:>8d} {a.status.value:12s}")


# -------------------------------------------------------------------
# dataset create
# -------------------------------------------------------------------

@cmd_dataset.command(name="create")
@click.argument("name")
@click.option("--task", "-t", default="detection",
              help="Task type: detection / pose / segmentation / anomaly")
@click.option("--display-name", help="Human-readable name")
@click.option("--description", "-d", help="Description")
@click.option("--owner", help="Owner")
def dataset_create(name: str, task: str, display_name: str, description: str, owner: str):
    """创建一个新的数据集资产。"""
    registry = _get_registry()
    try:
        asset = registry.create(
            name=name,
            task=task,
            display_name=display_name or name.replace("_", " ").title(),
            description=description or "",
            owner=owner or "",
        )
        click.echo(f"[Dataset] Created: {asset.name}")
        click.echo(f"  Path: data/datasets/{asset.name}/")
        click.echo(f"  Task: {asset.task}")
        click.echo(f"  Use 'raven-train dataset version {asset.name} --create' to add data")
    except FileExistsError as e:
        click.echo(f"[Dataset] {e}")


# -------------------------------------------------------------------
# dataset show
# -------------------------------------------------------------------

@cmd_dataset.command(name="show")
@click.argument("name")
def dataset_show(name: str):
    """查看数据集详情。"""
    registry = _get_registry()
    asset = registry.get(name)
    if asset is None:
        click.echo(f"[Dataset] '{name}' not found.")
        return

    click.echo(f"Name:        {asset.name}")
    click.echo(f"Display:     {asset.display_name}")
    click.echo(f"Task:        {asset.task}")
    click.echo(f"Status:      {asset.status.value}")
    click.echo(f"Version:     {asset.current_version} ({asset.version_count} total)")
    click.echo(f"Images:      {asset.total_images}")
    click.echo(f"Annotations: {asset.total_annotations}")
    click.echo(f"Owner:       {asset.owner}")
    click.echo(f"Tags:        {', '.join(asset.tags) if asset.tags else '(none)'}")
    click.echo(f"Created:     {asset.created_at}")
    click.echo(f"Updated:     {asset.updated_at}")

    # Check if there are versions
    vdir = _get_asset_dir(name) / "versions"
    if vdir.exists():
        versions = sorted(d.name for d in vdir.iterdir() if d.is_dir())
        if versions:
            click.echo(f"\nVersions: {', '.join(versions)}")


# -------------------------------------------------------------------
# dataset version
# -------------------------------------------------------------------

@cmd_dataset.command(name="version")
@click.argument("name")
@click.option("--create", is_flag=True, help="Create new version from ingested data")
@click.option("--list", "list_versions", is_flag=True, help="List versions")
@click.option("--source", help="Source directory (images) for version creation")
def dataset_version_cmd(name: str, create: bool, list_versions: bool, source: str):
    """管理数据集版本。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found. Create it first with 'raven-train dataset create {name}'")
        return

    vm = VersionManager(asset_dir)

    if list_versions:
        versions = vm.list_versions()
        if not versions:
            click.echo(f"[Dataset] No versions for '{name}'.")
            return
        click.echo(f"{'VERSION':10s} {'IMAGES':>8s} {'CREATED':22s} {'DESCRIPTION'}")
        click.echo("-" * 70)
        for v in versions:
            click.echo(f"{v.version:10s} {v.image_count:>8d} {v.created_at[:19]:22s} {v.description[:40]}")

    elif create:
        source_dir = Path(source) if source else R.dataset_dir(name)
        if not source_dir.exists():
            click.echo(f"[Dataset] Source directory not found: {source_dir}")
            click.echo("  Run 'raven-train ingest run data/raw' first, or specify --source")
            return

        snapshot = vm.create_version(
            source_images=source_dir,
            description=f"Auto-created from {source_dir}",
        )
        click.echo(f"[Dataset] Created version {snapshot.version} for '{name}'")
        click.echo(f"  Images: {snapshot.image_count}")
        click.echo(f"  Path: data/datasets/{name}/versions/{snapshot.version}/")

        # Update registry
        registry = _get_registry()
        asset = registry.get(name)
        if asset:
            asset.current_version = snapshot.version
            asset.version_count = len(vm.list_versions())
            asset.total_images = snapshot.image_count
            registry.update(asset)
            click.echo(f"  Registry updated: current_version={snapshot.version}")

    else:
        click.echo("Use --create or --list. See 'raven-train dataset version --help'")


# -------------------------------------------------------------------
# dataset stats
# -------------------------------------------------------------------

@cmd_dataset.command(name="stats")
@click.argument("name")
def dataset_stats(name: str):
    """查看数据集统计信息。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found.")
        return

    with MetadataStore(asset_dir / "metadata") as ms:
        stats = ms.get_statistics()
    if not stats:
        click.echo(f"[Dataset] No metadata for '{name}'. Add data first.")
        return

    click.echo(f"Dataset: {name}")
    click.echo(f"  Total images:     {stats.get('total_images', 0)}")
    click.echo(f"  Annotated:        {stats.get('annotated_count', 0)}")
    click.echo(f"  Cameras:          {', '.join(stats.get('cameras', [])) or '(none)'}")
    click.echo(f"  Shifts:           {', '.join(stats.get('shifts', [])) or '(none)'}")
    click.echo(f"  Scenes:           {', '.join(stats.get('scenes', [])) or '(none)'}")
    click.echo(f"  Resolutions:      {', '.join(stats.get('resolutions', [])) or '(none)'}")
    click.echo(f"  Lighting:         {', '.join(stats.get('lighting_conditions', [])) or '(none)'}")
    avg_q = stats.get("avg_quality_score", 0)
    click.echo(f"  Avg quality:      {avg_q:.1f}")


# -------------------------------------------------------------------
# dataset tag
# -------------------------------------------------------------------

@cmd_dataset.command(name="tag")
@click.argument("name")
@click.option("--add", help="Add a tag (e.g. 'night' 'occlusion' 'hard_example')")
@click.option("--category", default="general", help="Tag category")
@click.option("--list", "list_tags", is_flag=True, help="List all tags")
def dataset_tag(name: str, add: str, category: str, list_tags: bool):
    """管理数据集标签。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found.")
        return

    tm = TagManager(asset_dir)

    if list_tags:
        tags = tm.list_tags()
        if not tags:
            click.echo(f"[Dataset] No tags defined for '{name}'.")
            return
        click.echo(f"{'TAG':20s} {'CATEGORY':15s} {'DESCRIPTION'}")
        click.echo("-" * 60)
        for t in tags:
            click.echo(f"{t.name:20s} {t.category:15s} {t.description}")

    elif add:
        tm.define(name=add, category=category)
        click.echo(f"[Dataset] Tag '{add}' added to '{name}' (category: {category})")

    else:
        click.echo("Use --add or --list. See 'raven-train dataset tag --help'")


# -------------------------------------------------------------------
# dataset recipe
# -------------------------------------------------------------------

@cmd_dataset.command(name="recipe")
@click.argument("name")
@click.option("--create", is_flag=True, help="Create a new recipe")
@click.option("--list", "list_recipes", is_flag=True, help="List recipes")
@click.option("--show", help="Show a specific recipe")
@click.option("--build", is_flag=True, help="Build workspace from recipe")
@click.option("--output", "-o", default=None, help="Output workspace directory")
def dataset_recipe_cmd(name: str, create: bool, list_recipes: bool, show: str, build: bool, output: str):
    """管理数据集 Recipe。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found.")
        return

    rm = RecipeManager(asset_dir)

    if list_recipes:
        recipes = rm.list_recipes()
        if not recipes:
            click.echo(f"[Dataset] No recipes for '{name}'.")
            return
        click.echo("Recipes:")
        for r in recipes:
            click.echo(f"  {r.name}")

    elif show:
        recipe = rm.get_recipe(show)
        if recipe is None:
            click.echo(f"[Dataset] Recipe '{show}' not found.")
            return
        import yaml
        click.echo(yaml.dump(recipe.to_dict(), default_flow_style=False, sort_keys=False))

    elif create:
        recipe = DatasetRecipe(
            name=name,
            description=f"Default recipe for {name}",
        )
        path = rm.save_recipe(recipe)
        click.echo(f"[Dataset] Recipe created: {path}")

    elif build:
        """Build training workspace from Recipe."""
        from ..dataset_pipeline import DatasetPipeline
        from ..data_asset.profile import BuildProfile

        out_dir = Path(output) if output else None
        profile = BuildProfile.detection_default(name=name)
        pipeline = DatasetPipeline()
        try:
            result = pipeline.run_build(
                dataset_name=name,
                recipe_name=show or "default",
                profile=profile,
                output_dir=out_dir,
                force=False,
            )
            click.echo("[Dataset] Build complete:")
            click.echo(f"  Train: {result.train_count}  Val: {result.val_count}  Test: {result.test_count}")
            click.echo(f"  Classes: {result.class_count}")
            click.echo(f"  dataset.yaml: {result.dataset_yaml}")
            click.echo(f"  Output: {result.output_dir}")
            click.echo(f"  Use 'raven-train train {result.dataset_yaml}' to start training")
        except FileNotFoundError as e:
            click.echo(f"[Dataset] {e}")
        except FileExistsError:
            click.echo("[Dataset] Workspace already exists. Use --force to overwrite.")
        except RuntimeError as e:
            click.echo(f"[Dataset] Build failed: {e}")

    else:
        click.echo("Use --create, --list, --show, or --build.")


# -------------------------------------------------------------------
# dataset doctor
# -------------------------------------------------------------------

@cmd_dataset.command(name="doctor")
@click.argument("name")
def dataset_doctor(name: str):
    """数据集健康检查。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found.")
        return

    # Collect metadata stats
    with MetadataStore(asset_dir / "metadata") as ms:
        meta_count = ms.count()

    # Collect source distribution from registry
    registry = _get_registry()
    asset = registry.get(name)
    source_dist = {}
    if asset:
        vm = VersionManager(asset_dir)
        versions = vm.list_versions()
        if versions:
            latest = versions[0]
            source_dist = {latest.source: latest.image_count} if latest.source else {}

    # Check for manifest and annotations
    from ..data_asset.version import VersionManager as _VM
    vm = _VM(asset_dir)
    versions = vm.list_versions()
    manifest_path = None
    ann_dir = None
    if versions:
        latest = versions[0]
        manifest_path = R.dataset_manifest_path(name, latest.version)
        ann_dir = R.dataset_version_dir(name, latest.version) / "annotations" / "converted" / "yolo"

    doctor = DatasetDoctor(asset_dir)
    report = doctor.check(
        metadata_count=meta_count,
        source_distribution=source_dist if source_dist else None,
        manifest_path=manifest_path,
        ann_dir=ann_dir if ann_dir and ann_dir.exists() else None,
    )
    doctor.save_report(report)

    click.echo(f"\n{'='*50}")
    click.echo(f"  Dataset Doctor Report: {name}")
    click.echo(f"{'='*50}")
    click.echo(f"  Health Score: {report.overall_score:.0f}/100")
    click.echo(f"  Summary: {report.summary}")
    click.echo()

    for c in report.checks:
        status_icon = "✓" if c.status == "pass" else ("!" if c.status == "warn" else "✗")
        click.echo(f"  {status_icon} [{c.status.upper():5s}] {c.name:20s} {c.message}")

    if report.recommendations:
        click.echo("\n  Recommendations:")
        for r in report.recommendations:
            click.echo(f"    → {r}")

    click.echo()
    click.echo(f"  Full report: data/datasets/{name}/reports/health_latest.json")


# -------------------------------------------------------------------
# dataset lineage
# -------------------------------------------------------------------

@cmd_dataset.command(name="lineage")
@click.argument("name")
def dataset_lineage(name: str):
    """查看数据集血缘关系。"""
    asset_dir = _get_asset_dir(name)
    if not asset_dir.exists():
        click.echo(f"[Dataset] '{name}' not found.")
        return

    ls = LineageStore(asset_dir)
    chains = ls.get_full_chain()

    if not chains:
        click.echo(f"[Dataset] No lineage data for '{name}'.")
        return

    def _print_chain(chain, indent=0):
        prefix = "  " * indent
        click.echo(f"{prefix}• [{chain['type']}] {chain['name']} (id: {chain['id']})")
        for child in chain.get("children", []):
            click.echo(f"{prefix}  └─ {child['relation']}")
            _print_chain(child["node"], indent + 2)

    for chain in chains:
        _print_chain(chain)








# -------------------------------------------------------------------
# dataset split
# -------------------------------------------------------------------


@cmd_dataset.command(name="annotate")
@click.argument("name")
@click.option("--stats", is_flag=True, help="Show annotation statistics")
@click.option("--validate", is_flag=True, help="Validate annotation files")
@click.option("--convert-to", help="Convert annotations to format (cvat, yolo)")
@click.option("--format", default="yolo", help="Annotation format to use")
def dataset_annotate(name: str, stats: bool, validate: bool, convert_to: str, format: str):
    """管理数据集标注。"""
    from ..data_asset.annotation import AnnotationManager
    from ..data_asset.version import VersionManager as _VM

    asset_dir = _get_asset_dir(name)
    vm = _VM(asset_dir)
    versions = vm.list_versions()
    if not versions:
        click.echo(f"[Annotate] No versions for '{name}'.")
        return

    latest = versions[0]
    am = AnnotationManager(name, latest.version)

    if stats:
        s = am.get_statistics(format)
        click.echo(f"[Annotate] Statistics ({name} {latest.version}):")
        click.echo(f"  Total images:     {s.get('total_images', 0)}")
        click.echo(f"  Annotated:        {s.get('annotated_images', 0)}")
        click.echo(f"  Empty:            {s.get('empty_images', 0)}")
        click.echo(f"  Total annotations: {s.get('total_annotations', 0)}")
        cd = s.get('class_distribution', {})
        if cd:
            click.echo("  Class distribution:")
            for cls_id, count in sorted(cd.items()):
                click.echo(f"    Class {cls_id}: {count}")

    if validate:
        issues = am.validate(format)
        if not issues:
            click.echo("[Annotate] No issues found. ✓")
        else:
            errors = [i for i in issues if i['severity'] == 'error']
            warnings = [i for i in issues if i['severity'] == 'warn']
            click.echo(f"[Annotate] Found {len(errors)} errors, {len(warnings)} warnings")
            for issue in issues[:10]:
                click.echo(f"  [{issue['severity'].upper()}] {issue['file']}: {issue['issue']}")

    if convert_to:
        count = am.convert(format, convert_to)
        click.echo(f"[Annotate] Converted {count} annotations: {format} → {convert_to}")



@cmd_dataset.command(name="analyze")
@click.argument("name")
@click.option("--save", is_flag=True, help="Save report to reports/")
def dataset_analyze(name: str, save: bool):
    """数据智能分析：类别/摄像头/时间/目标尺寸分布。"""
    from ..data_asset.intelligence import DatasetIntelligence
    from ..data_asset.version import VersionManager as _VM

    asset_dir = _get_asset_dir(name)
    vm = _VM(asset_dir)
    versions = vm.list_versions()
    if not versions:
        click.echo(f"[Analyze] No versions for '{name}'.")
        return

    latest = versions[0]
    di = DatasetIntelligence(name, latest.version)
    report = di.analyze()

    click.echo("")
    click.echo("  " + "=" * 50)
    click.echo(f"  Dataset Intelligence: {name} {latest.version}")
    click.echo("  " + "=" * 50)
    click.echo(f"  Images:      {report.total_images} ({report.annotated_images} annotated, {report.background_images} bg)")
    click.echo(f"  Annotations: {report.total_annotations}")
    click.echo(f"  Background:  {report.background_ratio:.1f}%")
    click.echo("")

    if report.class_distribution:
        click.echo("  Class Distribution (top 10):")
        for cls_id, count in sorted(report.class_distribution.items(), key=lambda x: -x[1])[:10]:
            click.echo(f"    Class {cls_id}: {count}")

    if report.camera_distribution:
        click.echo("  Camera Distribution:")
        for cam, count in sorted(report.camera_distribution.items(), key=lambda x: -x[1]):
            click.echo(f"    {cam}: {count}")

    if report.time_distribution:
        click.echo("  Time Distribution:")
        total_time = max(sum(report.time_distribution.values()), 1)
        for period, count in sorted(report.time_distribution.items(), key=lambda x: -x[1]):
            click.echo(f"    {period}: {count} ({count/total_time*100:.1f}%)")

    click.echo("  Object Size:")
    click.echo(f"    Small:  {report.small_object_ratio:.1f}%")
    click.echo(f"    Medium: {report.medium_object_ratio:.1f}%")
    click.echo(f"    Large:  {report.large_object_ratio:.1f}%")

    if report.bias_warnings:
        click.echo("  Bias Warnings:")
        for w in report.bias_warnings:
            click.echo(f"    ! {w}")

    if report.recommendations:
        click.echo("  Recommendations:")
        for r in report.recommendations:
            click.echo(f"    -> {r}")

    if save:
        p = di.save_report(report)
        click.echo(f"  Report saved: {p}")


@cmd_dataset.command(name="task")
@click.argument("name")
@click.option("--create", help="Create task: task_id,platform,name (comma separated)")
@click.option("--list", "list_tasks", is_flag=True, help="List all annotation tasks")
@click.option("--status", help="Update task status: pending/in_progress/completed/reviewed")
@click.option("--show", help="Show task details by task_id")
def dataset_task(name: str, create: str, list_tasks: bool, status: str, show: str):
    """管理 CVAT/Label Studio 标注任务。"""
    from ..data_asset.annotation.task import TaskManager, AnnotationTask
    from ..data_asset.version import VersionManager as _VM
    from ..config import R as _R

    asset_dir = _get_asset_dir(name)
    vm = _VM(asset_dir)
    versions = vm.list_versions()
    if not versions:
        click.echo(f"[Task] No versions for '{name}'.")
        return
    latest = versions[0]
    tm = TaskManager(name, latest.version)

    if create:
        parts = [p.strip() for p in create.split(",")]
        if len(parts) < 3:
            click.echo('[Task] Usage: --create "task_id,platform,name"')
            return
        task = AnnotationTask(task_id=parts[0], platform=parts[1], name=parts[2])
        tm.create_task(task)
        click.echo(f"[Task] Created: {parts[0]} ({parts[1]})")

    if list_tasks:
        tasks = tm.list_tasks()
        if not tasks:
            click.echo(f"[Task] No tasks for '{name} {latest.version}'.")
            return
        click.echo(f"{'ID':20s} {'PLATFORM':15s} {'STATUS':15s} {'NAME'}")
        click.echo("-" * 70)
        for t in tasks:
            click.echo(f"{t.task_id:20s} {t.platform:15s} {t.status:15s} {t.name[:30]}")

    if status and show:
        tm.update_status(show, status)
        click.echo(f"[Task] {show} status -> {status}")

    if show and not status:
        task_obj: Optional[AnnotationTask] = tm.get_task(show)
        if task_obj:
            click.echo(f"Task ID:   {task_obj.task_id}")
            click.echo(f"Platform:  {task_obj.platform}")
            click.echo(f"Name:      {task_obj.name}")
            click.echo(f"Status:    {task_obj.status}")
            click.echo(f"Created:   {task_obj.created_at[:19]}")
            if task_obj.completed_at:
                click.echo(f"Completed: {task_obj.completed_at[:19]}")
            click.echo(f"Images:    {task_obj.image_count}")
            if task_obj.notes:
                click.echo(f"Notes:     {task_obj.notes}")
        else:
            click.echo(f"[Task] '{show}' not found.")



@cmd_dataset.command(name="build")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--val-size", type=float, default=0.2, help="Validation set ratio")
@click.option("--test-size", type=float, default=0.0, help="Test set ratio")
@click.option("--seed", type=int, default=42, help="Random seed")
def dataset_build(name: str, output: str | None, val_size: float, test_size: float, seed: int):
    """摄像头感知的数据集划分。"""
    from ..data_asset.split import DatasetSplitter

    asset_dir = _get_asset_dir(name)
    manifest_path = R.dataset_manifest_path(name, "v1")

    # Try to find the latest version with annotations
    from ..data_asset.version import VersionManager
    vm = VersionManager(asset_dir)
    versions = vm.list_versions()
    if not versions:
        click.echo(f"[Dataset] No versions found for '{name}'.")
        return

    latest = versions[0]
    manifest_path = R.dataset_manifest_path(name, latest.version)
    ann_dir = R.dataset_version_dir(name, latest.version) / "annotations" / "converted" / "yolo"

    if not manifest_path.exists():
        click.echo(f"[Dataset] Manifest not found: {manifest_path}")
        return
    if not ann_dir.exists():
        click.echo(f"[Dataset] Annotations not found: {ann_dir}")
        click.echo("  Run YOLO inference first or import annotations.")
        return

    out_dir = Path(output) if output else R.workspace_dir(f"{name}_split")
    splitter = DatasetSplitter(manifest_path, ann_dir)
    counts = splitter.split(out_dir, val_size=val_size, test_size=test_size, seed=seed)

    click.echo("[Dataset] Split complete:")
    click.echo(f"  Train: {counts.get('train', 0)}")
    click.echo(f"  Val:   {counts.get('val', 0)}")
    click.echo(f"  Test:  {counts.get('test', 0)}")
    click.echo(f"  Output: {out_dir}")

    # Generate dataset.yaml for YOLO
    import yaml
    from ..config import R as _R
    dataset_yaml: Dict[str, Any] = {
        "path": str(out_dir.resolve().as_posix()),
        "train": "train/images",
        "val": "val/images",
    }
    if test_size > 0:
        dataset_yaml["test"] = "test/images"
    # Get class count from annotations
    nc = 0
    for lbl in ann_dir.glob("*.txt"):
        if lbl.stat().st_size > 0:
            with open(lbl, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cls_id = int(line.split()[0])
                        nc = max(nc, cls_id + 1)
    dataset_yaml["nc"] = nc
    dataset_yaml["names"] = {i: str(i) for i in range(nc)}

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    click.echo(f"  dataset.yaml: {yaml_path}")
    click.echo(f"  Use this for training with: raven-train train {yaml_path}")


# -------------------------------------------------------------------
# dataset misdata
# -------------------------------------------------------------------

@cmd_dataset.command(name="misdata")
@click.argument("name")
@click.option("--collect", help="Collect misdata images from directory")
@click.option("--analyze", help="Analyze misdata with model (.pt)")
@click.option("--merge", is_flag=True, help="Merge misdata into dataset as new version")
@click.option("--review", is_flag=True, help="List all collected misdata images for review")
@click.option("--approve", help="Approve a misdata file by filename")
@click.option("--reject", help="Reject a misdata file by filename")
def dataset_misdata(name: str, collect: str, analyze: str, merge: bool, review: bool, approve: str, reject: str):
    """管理误报数据。"""
    from ..data_asset.misdata import MisdataManager

    mm = MisdataManager(name)

    if collect:
        count = mm.collect(collect)
        click.echo(f"[Misdata] Collected {count} images from {collect}")
        click.echo(f"  Stored in: {mm.misdata_dir}")

    if analyze:
        stats = mm.analyze(analyze)
        click.echo("[Misdata] Analysis results:")
        click.echo(f"  Total images: {stats.get('total_images', 0)}")
        click.echo(f"  Images with detections: {stats.get('images_with_detections', 0)}")
        by_class = stats.get("by_class", {})
        if by_class:
            click.echo("  Detections by class:")
            for label, count in sorted(by_class.items(), key=lambda x: -x[1]):
                click.echo(f"    {label}: {count}")

    if merge:
        count = mm.merge()
        click.echo(f"[Misdata] Merged {count} misdata images into '{name}'")
        click.echo(f"  Use 'raven-train dataset show {name}' to verify")
