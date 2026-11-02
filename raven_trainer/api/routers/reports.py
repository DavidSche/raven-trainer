"""Reports API — 报告查看器。"""

from __future__ import annotations

import re
from typing import List

from fastapi import APIRouter, HTTPException, Query

from ...config import R

router = APIRouter(tags=["reports"])

# Report ids are a single path component; anything else (wildcards, ``..``,
# separators) is rejected before it reaches a glob pattern.
_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")

REPORT_PATTERNS = [
    "**/train_report.html",
    "**/eval_report.html",
    "**/export_report.html",
    "**/intelligence_report.html",
    "**/ingestion_report.html",
    "**/health_report.html",
    "**/*_report.html",
]


@router.get("/reports")
def list_reports(limit: int = Query(50)):
    """Scan data/ and projects/ directories for HTML reports."""
    reports: List[dict] = []
    scan_dirs = [R.DATA, R.PROJECT_ROOT / "projects"] if R.PROJECT_ROOT.exists() else []
    for scan_root in scan_dirs:
        if not scan_root.exists():
            continue
        for pattern in REPORT_PATTERNS:
            for f in sorted(scan_root.glob(pattern))[:limit]:
                name = f.stem
                rel = f.relative_to(R.DATA) if f.is_relative_to(R.DATA) else f.relative_to(R.PROJECT_ROOT)
                reports.append({
                    "id": f"{f.parent.name}_{f.stem}",
                    "name": name,
                    "type": name.replace("_report", "").replace("_Report", ""),
                    "path": str(rel.as_posix()),
                    "created_at": f.stat().st_mtime,
                })
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return reports[:limit]


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    """Navigate to a report by finding it from path patterns."""
    if not _REPORT_ID_RE.match(report_id) or ".." in report_id:
        raise HTTPException(400, f"Invalid report id: {report_id!r}")
    scan_dirs = [R.DATA, R.PROJECT_ROOT / "projects"] if R.PROJECT_ROOT.exists() else [R.DATA]
    for scan_root in scan_dirs:
        if not scan_root.exists():
            continue
        for pattern in REPORT_PATTERNS:
            for f in scan_root.glob(pattern):
                rid = f"{f.parent.name}_{f.stem}"
                if rid == report_id:
                    try:
                        return {"html": f.read_text(encoding="utf-8")}
                    except OSError as exc:
                        raise HTTPException(404, f"Report '{report_id}' unreadable") from exc
    # Try to find by name in dataset report directories
    datasets_dir = R.DATASETS
    if datasets_dir.exists():
        for ds_dir in datasets_dir.iterdir():
            if ds_dir.is_dir():
                reports_dir = ds_dir / "reports"
                if reports_dir.exists():
                    for rf in reports_dir.glob(f"*{report_id.replace('intelligence_', '')}*"):
                        if rf.exists() and rf.is_file():
                            try:
                                if rf.suffix == ".html":
                                    return {"html": rf.read_text(encoding="utf-8")}
                                return {"html": f"<pre>{rf.read_text(encoding='utf-8')}</pre>"}
                            except OSError as exc:
                                raise HTTPException(404, f"Report '{report_id}' unreadable") from exc
    raise HTTPException(404, f"Report '{report_id}' not found")
