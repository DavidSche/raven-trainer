"""FastAPI server for Raven Trainer Web UI.

Started via `raven-train ui` CLI command.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import R
from .routers import datasets, training, models, job, ingestion, reports, system, tasks, feedback, pipeline, recipes
from .routers import storage as storage_router

logger = logging.getLogger("raven-trainer.api")

app = FastAPI(
    title="Raven Trainer API",
    version="0.1.0",
    description="Raven Trainer — Industrial Vision Data Asset & Training Platform",
)

# CORS: allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes (registered before UI catch-all to take priority)
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(job.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(pipeline.router, prefix="/api/v1")
app.include_router(storage_router.router, prefix="/api/v1")
app.include_router(recipes.router, prefix="/api/v1")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.on_event("startup")
async def startup():
    R.ensure_dirs()
    # Capture the main event loop so background threads in training.py can
    # schedule WebSocket broadcasts via run_coroutine_threadsafe (Python 3.12+
    # no longer returns a usable loop from get_event_loop() in non-main threads).
    import asyncio as _asyncio
    from .routers.training import set_main_loop
    set_main_loop(_asyncio.get_running_loop())

    # Recover from ungraceful shutdown: any job still marked RUNNING when the
    # process restarts was interrupted by the restart (daemon threads are killed
    # without finally blocks). Mark them FAILED so they don't stay as zombies.
    try:
        from .routers import job as _job_module  # noqa: F401
        from ..data_asset.job import JobManager, JobStatus
        jm = JobManager()
        for job in jm.list_jobs(limit=1000):
            if job.status == JobStatus.RUNNING:
                jm.update_status(
                    job.job_id,
                    JobStatus.FAILED,
                    error="interrupted by process restart",
                )
                try:
                    jm.append_log(job.job_id, "[Recovery] Marked FAILED due to process restart.")
                except Exception:
                    pass
    except Exception:
        # Best-effort recovery; never block startup
        import traceback as _tb
        print(f"[startup] Job recovery skipped: {_tb.format_exc()}")


# Serve built UI (SPA fallback: serve index.html for all non-API paths)
UI_DIST = Path(__file__).resolve().parent.parent.parent / "ui_src" / "dist"


async def _serve_ui(path: str = ""):
    if UI_DIST.exists():
        file = UI_DIST / path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        index = UI_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
    from fastapi.responses import JSONResponse
    return JSONResponse({"status": "ui-not-built", "message": "Run 'npm run build' in ui_src/"}, status_code=200)


@app.get("/")
async def root():
    return await _serve_ui("")


@app.get("/assets/{path:path}")
async def serve_assets(path: str):
    return await _serve_ui(f"assets/{path}")


@app.get("/{path:path}")
async def serve_spa(path: str):
    # Don't catch API paths
    if path.startswith("api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return await _serve_ui(path)


def run_server(port: int = 8081, host: str = "127.0.0.1", open_browser: bool = False):
    """Run the FastAPI dev server."""
    import os
    import uvicorn
    logger.info(f"Starting Raven Trainer UI at http://{host}:{port}")
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://{host}:{port}")
    reload_enabled = os.environ.get("RAVEN_RELOAD", "1") != "0"
    if reload_enabled:
        # When reload=True, uvicorn requires the app as an import string
        uvicorn.run("raven_trainer.api.server:app", host=host, port=port, log_level="info", reload=True)
    else:
        uvicorn.run(app, host=host, port=port, log_level="info", reload=False)
