"""Job — 统一任务系统。

为 CLI 和未来 Web UI 提供统一的任务运行、状态跟踪、日志记录。
每个操作（train/eval/export/ingest）都是一个 Job。

Job 生命周期：
  pending → running → completed | failed | cancelled

存储：
  data/jobs/<job_id>/
    job.yaml          ← Job 配置和状态
    log.txt           ← 运行日志
    result.yaml       ← 输出产物

Usage:
    job = Job.create("train", {"epochs": 100, "model": "yolo26n"})
    job.run()          # 同步执行
    job.status         # running / completed
    job.log("Training epoch 1...")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...config import R

# ``job_id`` is used to build filesystem paths; only accept IDs this module can
# create (``<job_type>_<4+ digits>``) so API input cannot escape ``data/jobs``.
_JOB_ID_RE = re.compile(r"^(?:ingest|train|eval|export|build|pipeline)_\d{4,}$")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    INGEST = "ingest"
    TRAIN = "train"
    EVAL = "eval"
    EXPORT = "export"
    BUILD = "build"
    PIPELINE = "pipeline"


@dataclass
class Job:
    """统一任务记录。"""

    job_id: str
    type: JobType
    config: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    log_path: str = ""
    result_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "type": self.type.value,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_path": self.log_path,
            "result_path": self.result_path,
            "error": self.error,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Job":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["type"] = JobType(data["type"])
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobManager:
    """统一任务管理器。

    管理 data/jobs/ 目录下的所有 Job。
    CLI 命令和 future Web UI 都通过 JobManager 启动和跟踪任务。
    """

    def __init__(self):
        self.jobs_dir = R.DATA / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        job_type: JobType,
        config: Optional[Dict[str, Any]] = None,
        label: str = "",
    ) -> Job:
        """创建新任务。

        Job ID 序号基于该类型已有最大序号 +1（而非 len(existing)+1），
        避免删除任务后再创建时与历史 ID 冲突。
        """
        prefix = f"{job_type.value}_"
        max_seq = 0
        if self.jobs_dir.exists():
            for f in self.jobs_dir.glob(f"{prefix}*.json"):
                # Skip *_result.json sidecars
                if f.stem.endswith("_result"):
                    continue
                m = re.match(rf"^{re.escape(prefix)}(\d+)\.json$", f.name)
                if m:
                    seq_num = int(m.group(1))
                    if seq_num > max_seq:
                        max_seq = seq_num
        seq = max_seq + 1
        job_id = f"{prefix}{seq:04d}"
        # Guard against the (very unlikely) race where the file was created
        # between our scan and save. If it exists, bump until we find a free slot.
        job_path = self.jobs_dir / f"{job_id}.json"
        while job_path.exists():
            seq += 1
            job_id = f"{prefix}{seq:04d}"
            job_path = self.jobs_dir / f"{job_id}.json"
        job = Job(
            job_id=job_id,
            type=job_type,
            config=config or {},
        )
        job.created_at = datetime.now().isoformat()
        job.save(job_path)

        # Create log and result paths
        log_path = self.jobs_dir / f"{job_id}_log.txt"
        result_path = self.jobs_dir / f"{job_id}_result.json"
        job.log_path = str(log_path)
        job.result_path = str(result_path)
        log_path.write_text(f"[{datetime.now().isoformat()}] Job {job_id} created\n", encoding="utf-8")
        result_path.write_text("{}\n", encoding="utf-8")
        job.save(job_path)

        return job

    def get(self, job_id: str) -> Optional[Job]:
        if not _JOB_ID_RE.match(job_id or ""):
            return None
        path = self.jobs_dir / f"{job_id}.json"
        if path.exists():
            return Job.load(path)
        return None

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> List[Job]:
        jobs = []
        # Filter *before* applying the limit, otherwise corrupt/unrelated JSON
        # files (or a status filter) would silently shrink the result set.
        for f in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            # Skip result/log files — they match *.json but aren't Job records
            if f.stem.endswith("_result") or f.stem.endswith("_log"):
                continue
            if not _JOB_ID_RE.match(f.stem):
                continue
            try:
                job = Job.load(f)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"  [WARN] Skipping corrupt job file {f.name}: {e}")
                continue
            if status is None or job.status == status:
                jobs.append(job)
                if len(jobs) >= limit:
                    break
        return jobs

    def update_status(self, job_id: str, status: JobStatus, error: str = "") -> Job:
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = status
        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.now().isoformat()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.finished_at = datetime.now().isoformat()
        if error:
            job.error = error
        job.save(self.jobs_dir / f"{job_id}.json")
        return job

    def append_log(self, job_id: str, message: str) -> None:
        job = self.get(job_id)
        if job and job.log_path:
            with open(job.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def save_result(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self.get(job_id)
        if job and job.result_path:
            with open(job.result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
