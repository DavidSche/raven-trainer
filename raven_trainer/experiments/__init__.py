"""Experiments — 训练实验管理。

一个 Experiment 记录一次完整的训练过程。
每个实验包含：
  - 配置: 模型、参数、数据集
  - 可复现元数据: git_commit, adapter_version, torch_version
  - 结果: 指标、模型路径
  - 产物: .pt, .onnx, metadata

存储路径:
  projects/<project>/experiments/
    exp_001/
      experiment.yaml
      metrics.json
      train_report.html
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import R, validate_name

# Serialises experiment-id allocation within this process so two concurrent
# training starts cannot both pick the same sequence number.
_EXPERIMENT_ID_LOCK = threading.Lock()


@dataclass
class Experiment:
    """A single training experiment."""
    experiment_id: str
    project: str
    model: str
    backend: str = "ultralytics"
    status: str = "created"  # created / running / completed / failed
    epochs: int = 0
    batch: int = 0
    imgsz: int = 0
    device: str = "cuda:0"
    dataset: str = ""
    workspace: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    best_model_path: str = ""
    onnx_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "project": self.project,
            "model": self.model,
            "backend": self.backend,
            "status": self.status,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "dataset": self.dataset,
            "workspace": self.workspace,
            "config": self.config,
            "metrics": self.metrics,
            "reproducibility": self.reproducibility,
            "created_at": self.created_at or datetime.now().isoformat(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "best_model_path": self.best_model_path,
            "onnx_path": self.onnx_path,
            "error": self.error,
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "experiment.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "Experiment":
        with open(path / "experiment.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class ExperimentManager:
    """Manages experiments for a project."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.exp_root = R.project_dir(project_name) / "experiments"

    def _exp_dir(self, experiment_id: str) -> Path:
        """Resolve an experiment directory, rejecting unsafe ids."""
        return self.exp_root / validate_name(experiment_id, "experiment id")

    def create(self, **kwargs) -> Experiment:
        with _EXPERIMENT_ID_LOCK:
            # Sequence based on max existing number rather than len(existing),
            # so deleting an experiment and creating a new one does not collide
            # with a previously-deleted higher-numbered experiment's records.
            max_seq = 0
            if self.exp_root.exists():
                for d in self.exp_root.iterdir():
                    if not d.is_dir():
                        continue
                    m = re.match(r"^exp_(\d+)$", d.name)
                    if m:
                        n = int(m.group(1))
                        if n > max_seq:
                            max_seq = n
            seq = max_seq + 1
            exp = Experiment(
                experiment_id=f"exp_{seq:04d}",
                project=self.project_name,
                created_at=datetime.now().isoformat(),
                **kwargs,
            )
            exp.save(self.exp_root / exp.experiment_id)
            return exp

    def get(self, experiment_id: str) -> Optional[Experiment]:
        path = self._exp_dir(experiment_id)
        if path.exists() and (path / "experiment.yaml").exists():
            return Experiment.load(path)
        return None

    def list(self) -> List[Experiment]:
        results: List[Experiment] = []
        if not self.exp_root.exists():
            return results
        for d in sorted(self.exp_root.iterdir(), reverse=True):
            if d.is_dir() and (d / "experiment.yaml").exists():
                try:
                    results.append(Experiment.load(d))
                except Exception:
                    pass
        return results

    def update_status(self, experiment_id: str, status: str, error: str = "") -> Optional[Experiment]:
        exp = self.get(experiment_id)
        if exp:
            exp.status = status
            if status == "running" and not exp.started_at:
                exp.started_at = datetime.now().isoformat()
            if status in ("completed", "failed"):
                exp.finished_at = datetime.now().isoformat()
            if error:
                exp.error = error
            exp.save(self._exp_dir(experiment_id))
        return exp

    def update_metrics(self, experiment_id: str, metrics: dict) -> Optional[Experiment]:
        exp = self.get(experiment_id)
        if exp:
            exp.metrics = metrics
            exp.save(self._exp_dir(experiment_id))
        return exp



    def append_log(self, experiment_id: str, message: str) -> None:
        """Append a log line to the experiments log file."""
        log_dir = self._exp_dir(experiment_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train.log"
        if log_path.is_dir():
            import shutil
            shutil.rmtree(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def get_log(self, experiment_id: str) -> str:
        """Get the full log content."""
        log_path = self._exp_dir(experiment_id) / "train.log"
        if log_path.exists():
            return log_path.read_text(encoding="utf-8")
        return ""

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment directory. Returns True if something was removed."""
        import shutil

        exp_dir = self._exp_dir(experiment_id)
        if not exp_dir.exists():
            return False
        shutil.rmtree(exp_dir)
        return True

# Factory function

def get_manager_for_project(project_name: str) -> ExperimentManager:
    return ExperimentManager(project_name)


def scan_all_experiments() -> List[Experiment]:
    """Scan all projects for experiments."""
    results: List[Experiment] = []
    projects_dir = R.PROJECTS
    if not projects_dir.exists():
        return results
    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        mgr = ExperimentManager(proj_dir.name)
        try:
            results.extend(mgr.list())
        except Exception:
            pass
    return results
