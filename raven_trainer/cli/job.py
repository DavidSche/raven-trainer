"""raven-train job — 统一任务管理。

Usage:
    raven-train job list              List all jobs
    raven-train job show <id>         Show job details
    raven-train job log <id>          Show job log
"""

from __future__ import annotations

from pathlib import Path

import click

from ..data_asset.job import JobManager, JobStatus


@click.group(name="job")
def cmd_job():
    """统一任务管理（Job System）。"""


@cmd_job.command(name="list")
@click.option("--status", default=None, help="Filter by status: pending/running/completed/failed")
@click.option("--limit", type=int, default=20, help="Max jobs to show")
def job_list(status: str | None, limit: int):
    """列出所有任务。"""
    jm = JobManager()
    status_filter = JobStatus(status) if status else None
    jobs = jm.list_jobs(status=status_filter, limit=limit)
    if not jobs:
        click.echo("[Job] No jobs found.")
        return
    click.echo(f"{'ID':20s} {'TYPE':10s} {'STATUS':12s} {'CREATED':22s}")
    click.echo("-" * 64)
    for j in jobs:
        click.echo(f"{j.job_id:20s} {j.type.value:10s} {j.status.value:12s} {j.created_at[:19]:22s}")


@cmd_job.command(name="show")
@click.argument("job_id")
def job_show(job_id: str):
    """查看任务详情。"""
    jm = JobManager()
    job = jm.get(job_id)
    if job is None:
        click.echo(f"[Job] '{job_id}' not found.")
        return
    click.echo(f"ID:       {job.job_id}")
    click.echo(f"Type:     {job.type.value}")
    click.echo(f"Status:   {job.status.value}")
    click.echo(f"Created:  {job.created_at[:19]}")
    if job.started_at:
        click.echo(f"Started:  {job.started_at[:19]}")
    if job.finished_at:
        click.echo(f"Finished: {job.finished_at[:19]}")
    if job.error:
        click.echo(f"Error:    {job.error}")
    if job.config:
        import yaml as _yaml
        click.echo(f"Config:\n{_yaml.dump(job.config, default_flow_style=False)}")


@cmd_job.command(name="log")
@click.argument("job_id")
def job_log(job_id: str):
    """查看任务日志。"""
    jm = JobManager()
    job = jm.get(job_id)
    if job is None:
        click.echo(f"[Job] '{job_id}' not found.")
        return
    if job.log_path and Path(job.log_path).exists():
        click.echo(Path(job.log_path).read_text(encoding="utf-8"))
    else:
        click.echo("[Job] No log available.")
