"""Job API — 统一任务管理。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..schemas import JobSchema, JobLogSchema
from ...data_asset.job import JobManager, JobStatus, JobType

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None),
    type_filter: Optional[str] = Query(None, alias="type"),
    dataset_name: Optional[str] = Query(None, alias="dataset_name"),
    limit: int = Query(50),
):
    jm = JobManager()
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status filter: {status}")
    jobs = jm.list_jobs(status=status_filter, limit=limit)
    if type_filter:
        jobs = [j for j in jobs if (j.type.value if hasattr(j.type, 'value') else j.type) == type_filter]
    if dataset_name:
        jobs = [j for j in jobs if j.config and j.config.get('dataset_name') == dataset_name]
    return [JobSchema(
        job_id=j.job_id,
        type=j.type.value if hasattr(j.type, 'value') else j.type,
        status=j.status.value if hasattr(j.status, 'value') else j.status,
        created_at=j.created_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
        error=j.error,
        config=j.config,
    ) for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    jm = JobManager()
    job = jm.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JobSchema(
        job_id=job.job_id,
        type=job.type.value if hasattr(job.type, 'value') else job.type,
        status=job.status.value if hasattr(job.status, 'value') else job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        config=job.config,
    )


@router.get("/jobs/{job_id}/log")
async def get_job_log(job_id: str):
    from pathlib import Path
    jm = JobManager()
    job = jm.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if job.log_path and Path(job.log_path).exists():
        log = Path(job.log_path).read_text(encoding="utf-8")
        return JobLogSchema(job_id=job_id, log=log)
    return JobLogSchema(job_id=job_id, log="")


# ── WebSocket for real-time logs ──

@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    jm = JobManager()
    # M-27: Track the byte offset already streamed so we only read the
    # increment appended since the last poll, instead of re-reading the
    # entire log file every 2s (which scales linearly with log size).
    last_len = 0
    # Pre-seed last_len to the current log size so the client doesn't get
    # flooded with the full history on the first frame.
    first_job = jm.get(job_id)
    if first_job and first_job.log_path:
        from pathlib import Path
        _lp = Path(first_job.log_path)
        if _lp.exists():
            last_len = _lp.stat().st_size
    try:
        while True:
            job = jm.get(job_id)
            if job:
                incremental = ""
                if job.log_path:
                    from pathlib import Path
                    log_path = Path(job.log_path)
                    if log_path.exists():
                        size = log_path.stat().st_size
                        if size > last_len:
                            with open(log_path, "r", encoding="utf-8") as f:
                                f.seek(last_len)
                                incremental = f.read()
                            last_len = size
                status_value = job.status.value if hasattr(job.status, 'value') else job.status
                await websocket.send_json({
                    "status": status_value,
                    "log": incremental,
                })
                # Exit the loop once the job has reached a terminal state
                # (COMPLETED / FAILED / CANCELLED) so the websocket closes
                # instead of polling forever.
                if status_value in ("completed", "failed", "cancelled"):
                    break
            import asyncio
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
