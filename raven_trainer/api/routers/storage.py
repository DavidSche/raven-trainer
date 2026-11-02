"""Storage API — 原始数据层（Raw, Immutable）管理。

展示 data/storage/ 根目录下的所有媒体类型（images, videos），
支持浏览、视频转图片抽取、闭环数据集创建。

存储规范：
  storage/
    images/       ← 原始图片，按来源子目录组织
    videos/       ← 原始视频，按来源子目录组织

每个媒体类型下可以有多个"源（source）"子目录。
数据集通过 manifest.jsonl 的 storage_path 引用 storage/ 中的文件。
"""

from __future__ import annotations

import json as _json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from ...config import R
from ...data_asset.changelog import ChangelogStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["storage"])

MEDIA_TYPES = {
    "images": {
        "label": "Images",
        "extensions": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
    },
    "videos": {
        "label": "Videos",
        "extensions": {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"},
    },
}

def _list_media_type_dirs(media_type: str) -> list[Path]:
    base = R.STORAGE / media_type
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name)

# ---------------------------------------------------------------------------
# M-20: Reverse index for storage_path → [(dataset_name, version)]
# Built once on first call or when TTL expires; rebuilt on server startup.
# ---------------------------------------------------------------------------
_storage_index: Dict[str, List[Tuple[str, str]]] = {}
_storage_index_ts: float = 0.0
_STORAGE_INDEX_TTL: float = 300.0  # 5 minutes


def _build_storage_index() -> Dict[str, List[Tuple[str, str]]]:
    """Scan all manifest.jsonl files and build a reverse index:
    storage_path → [(dataset_name, version)].
    """
    index: Dict[str, List[Tuple[str, str]]] = {}
    datasets_dir = R.DATASETS
    if not datasets_dir.exists():
        return index
    for ds_dir in sorted(datasets_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        versions_dir = ds_dir / "versions"
        if not versions_dir.exists():
            continue
        for ver_dir in sorted(versions_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            manifest_path = ver_dir / "manifest.jsonl"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                            sp = rec.get("storage_path", "")
                            if sp:
                                index.setdefault(sp, []).append(
                                    (ds_dir.name, ver_dir.name)
                                )
                        except _json.JSONDecodeError:
                            continue
            except Exception:
                continue
    return index


def _get_datasets_for_storage_path(fragment: str) -> list[dict]:
    global _storage_index, _storage_index_ts
    # Rebuild index if TTL expired
    if time.time() - _storage_index_ts > _STORAGE_INDEX_TTL:
        _storage_index = _build_storage_index()
        _storage_index_ts = time.time()
    # Search index for entries whose storage_path contains the fragment
    results = []
    seen: set = set()
    for sp, entries in _storage_index.items():
        if fragment in sp:
            for ds_name, ver in entries:
                key = (ds_name, ver)
                if key not in seen:
                    seen.add(key)
                    results.append({"dataset_name": ds_name, "version": ver})
    return results

def _get_media_info(path: Path, media_type: str) -> dict:
    stat = path.stat()
    info = {"filename": path.name, "size_bytes": stat.st_size,
            "size_display": _format_bytes(stat.st_size), "modified": stat.st_mtime}
    if media_type == "images":
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as img:
                info["width"], info["height"] = img.size
        except Exception:
            info["width"] = info["height"] = 0
    elif media_type == "videos":
        info.update({"width": 0, "height": 0, "duration_seconds": 0.0})
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                data = _json.loads(result.stdout)
                fmt = data.get("format", {})
                info["duration_seconds"] = float(fmt.get("duration", 0))
                info["format_name"] = fmt.get("format_name", "")
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info["width"] = int(stream.get("width", 0))
                        info["height"] = int(stream.get("height", 0))
                        info["codec"] = stream.get("codec_name", "")
                        info["fps"] = _parse_fps(stream.get("r_frame_rate", "0/1"))
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired, _json.JSONDecodeError):
            pass
    return info

def _parse_fps(s: str) -> float:
    try:
        parts = s.split("/")
        return float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0

def _format_bytes(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024: return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / (1024 * 1024 * 1024):.2f} GB"

def _format_duration(seconds: float) -> str:
    if seconds <= 0: return "0:00"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── Storage root ──

@router.get("/storage/sources/for-dataset/{dataset_name}")
async def get_sources_for_dataset(dataset_name: str):
    results = []
    for mt_name in MEDIA_TYPES:
        base = R.STORAGE / mt_name
        if not base.exists():
            continue
        for src_dir in sorted(base.iterdir()):
            if not src_dir.is_dir():
                continue
            ds_list = _get_datasets_for_storage_path(src_dir.name)
            if any(d["dataset_name"] == dataset_name for d in ds_list):
                files = [f for f in src_dir.iterdir() if f.is_file()
                         and f.suffix.lower() in MEDIA_TYPES[mt_name]["extensions"]]
                results.append({"source": src_dir.name, "media_type": mt_name, "file_count": len(files)})
    return {"dataset": dataset_name, "sources": results}


@router.get("/storage")
async def get_storage_root():
    media_types = []
    total_images = total_videos = total_size = 0
    for mt_name, mt_config in MEDIA_TYPES.items():
        sources = _list_media_type_dirs(mt_name)
        source_list = []
        type_files = type_size = 0
        for src_dir in sources:
            files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in mt_config["extensions"]]
            if not files:
                continue
            src_size = sum(f.stat().st_size for f in files)
            datasets = _get_datasets_for_storage_path(src_dir.name)
            preview = _get_media_info(files[0], mt_name)
            source_list.append({
                "name": src_dir.name, "display_name": src_dir.name.replace("-", " ").title(),
                "media_type": mt_name, "file_count": len(files), "total_size": src_size,
                "total_size_display": _format_bytes(src_size),
                "preview": preview, "datasets": datasets,
            })
            type_files += len(files)
            type_size += src_size
        if mt_name == "images": total_images += type_files
        else: total_videos += type_files
        total_size += type_size
        media_types.append({
            "media_type": mt_name, "label": mt_config["label"],
            "source_count": len(source_list), "file_count": type_files,
            "total_size": type_size, "total_size_display": _format_bytes(type_size),
            "sources": source_list,
        })
    return {
        "root_path": str(R.STORAGE),
        "media_types": media_types,
        "summary": {
            "total_images": total_images, "total_videos": total_videos,
            "total_size": total_size, "total_size_display": _format_bytes(total_size),
            "total_sources": sum(mt["source_count"] for mt in media_types),
        },
    }


@router.get("/storage/{media_type}")
async def get_media_type(media_type: str):
    if media_type not in MEDIA_TYPES:
        raise HTTPException(404, f"Unknown media type '{media_type}'")
    mt_config = MEDIA_TYPES[media_type]
    sources = _list_media_type_dirs(media_type)
    source_list = []
    for src_dir in sources:
        files = sorted([f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in mt_config["extensions"]],
                       key=lambda f: f.name)
        if not files:
            continue
        src_size = sum(f.stat().st_size for f in files)
        datasets = _get_datasets_for_storage_path(src_dir.name)
        previews = [_get_media_info(f, media_type) for f in files[:12]]
        source_list.append({
            "name": src_dir.name, "display_name": src_dir.name.replace("-", " ").title(),
            "file_count": len(files), "total_size": src_size,
            "total_size_display": _format_bytes(src_size),
            "datasets": datasets, "previews": previews,
        })
    return {
        "media_type": media_type, "label": mt_config["label"],
        "source_count": len(source_list),
        "file_count": sum(s["file_count"] for s in source_list),
        "sources": source_list,
    }


@router.get("/storage/{media_type}/{source}/browse")
async def browse_source(media_type: str, source: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    if media_type not in MEDIA_TYPES:
        raise HTTPException(404, f"Unknown media type '{media_type}'")
    src_dir = R.STORAGE / media_type / source
    if not src_dir.exists() or not src_dir.is_dir():
        raise HTTPException(404, f"Source '{source}' not found in {media_type}")
    exts = MEDIA_TYPES[media_type]["extensions"]
    all_files = sorted([f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in exts], key=lambda f: f.name)
    total = len(all_files)
    page = all_files[offset:offset + limit]
    return {"media_type": media_type, "source": source, "total": total, "offset": offset, "limit": limit,
            "files": [_get_media_info(f, media_type) for f in page]}


@router.get("/storage/{media_type}/{source}/file/{filename:path}")
async def serve_source_file(media_type: str, source: str, filename: str):
    if media_type not in MEDIA_TYPES:
        raise HTTPException(404, f"Unknown media type '{media_type}'")
    src_resolved = (R.STORAGE / media_type / source).resolve()
    file_path = (R.STORAGE / media_type / source / filename).resolve()
    # Use is_relative_to (Python 3.9+) instead of str.startswith to avoid
    # prefix-collision attacks (e.g. source 'foo' matching 'foobar/secret.jpg')
    if not file_path.is_relative_to(src_resolved):
        raise HTTPException(403, "Forbidden")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"File '{filename}' not found")
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp",
            ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(file_path), media_type=mime)


# ── Video info ──

@router.get("/storage/videos/{source}/info")
async def get_video_source_info(source: str):
    src_dir = R.STORAGE / "videos" / source
    if not src_dir.exists() or not src_dir.is_dir():
        raise HTTPException(404, f"Video source '{source}' not found")
    exts = MEDIA_TYPES["videos"]["extensions"]
    files = sorted([f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in exts], key=lambda f: f.name)
    files_info = [_get_media_info(f, "videos") for f in files]
    total_duration = sum(f.get("duration_seconds", 0) for f in files_info)
    total_size = sum(f.get("size_bytes", 0) for f in files_info)
    datasets = _get_datasets_for_storage_path(source)
    return {
        "source": source, "display_name": source.replace("-", " ").title(),
        "file_count": len(files_info), "total_duration_seconds": total_duration,
        "total_duration_display": _format_duration(total_duration),
        "total_size": total_size, "total_size_display": _format_bytes(total_size),
        "datasets": datasets, "files": files_info,
    }


# ── Video extraction ──

class ExtractFramesRequest(BaseModel):
    dataset_name: str = ""
    sampling: str = "interval"
    interval_seconds: float = 5.0
    max_frames: int = 500
    quality_min_score: float = 0.3
    dedup_threshold: float = 0.95
    # Decode config (new unified field, overrides sampling if provided)
    decode: Optional[dict] = None  # {mode, every_n_frames, target_fps, jpeg_quality}

    def to_profile(self, source_name: str) -> "IngestionProfile":
        """Build an IngestionProfile from this request — the single-source quick-action wrapper."""
        from ...ingestion.profile import IngestionProfile, SourceRef
        from ...ingestion.sampling.core import DecodeConfig

        # Build decode config
        if self.decode is not None:
            decode = dict(self.decode)  # use explicit decode config
            # Fill missing defaults
            decode.setdefault("mode", "interval")
            decode.setdefault("every_n_frames", 30)
            decode.setdefault("target_fps", 0.0)
            decode.setdefault("jpeg_quality", 80)
        else:
            # Legacy mapping: sampling string → decode mode
            mode_map = {
                "interval": "interval",
                "scene_change": "interval",  # scene change is a sampler, not a decode mode
                "motion": "interval",
                "burst": "interval",
            }
            decode = {
                "mode": mode_map.get(self.sampling, "interval"),
                # Default 25fps assumption when actual fps is unknown;
                # override by passing decode.target_fps or decode.every_n_frames
                "every_n_frames": max(1, int(self.interval_seconds * self.decode.get("target_fps", 25) if self.decode and self.decode.get("target_fps") else self.interval_seconds * 25)),
                "target_fps": 0.0,
                "jpeg_quality": 80,
            }

        # Pipeline: use the single legacy sampler if provided, otherwise use balance as minimal filter
        legacy_sampler = self.sampling if self.sampling in ("interval", "scene_change", "motion", "burst") else None
        pipeline = [legacy_sampler] if legacy_sampler else ["balance"]

        return IngestionProfile(
            name=f"_quick_extract_{source_name}",
            description=f"Quick extract from video source '{source_name}'",
            sources=[SourceRef(name=source_name, path=f"videos/{source_name}")],
            decode=decode,
            pipeline=pipeline,
            sampling={},
            sample_count=self.max_frames,
            # M-39: Previously quality={} and dedup={} silently dropped the
            # user-supplied quality_min_score / dedup_threshold from the
            # request, so the quick-extract endpoint always bypassed both
            # filters regardless of what the client sent. Wire the request
            # fields through to the profile so the thresholds actually take
            # effect.
            quality={
                "blur_laplacian_threshold": self.quality_min_score,
                "black_mean_threshold": 5.0,
                "overexposed_mean_threshold": 250.0,
                "overexposed_pixel_ratio": 0.5,
                "min_width": 320,
                "min_height": 240,
            },
            dedup={
                "method": "dhash",
                "hash_size": 8,
                "hash_distance": 5,
                "ssim_threshold": self.dedup_threshold,
            },
            auto_register=True,
            target_dataset=self.dataset_name or f"extracted_{source_name}",
        )


@router.post("/storage/videos/{source}/extract")
async def extract_frames_from_video(source: str, req: ExtractFramesRequest):
    src_dir = R.STORAGE / "videos" / source
    if not src_dir.exists() or not src_dir.is_dir():
        raise HTTPException(404, f"Video source '{source}' not found")
    exts = MEDIA_TYPES["videos"]["extensions"]
    videos = sorted([f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in exts], key=lambda f: f.name)
    if not videos:
        raise HTTPException(400, f"No video files found in source '{source}'")

    dataset_name = req.dataset_name or f"extracted_{source}"

    try:
        from ...ingestion.profile import IngestionProfileManager

        # Build temporary profile using the request's to_profile method
        profile = req.to_profile(source)

        # Run via IngestionProfileManager (same engine as IngestionProfilesPage).
        # run_profile(name) loads the profile from disk, so we MUST persist it first.
        mgr = IngestionProfileManager()
        mgr.save_profile(profile)
        try:
            report = mgr.run_profile(profile.name, output_dir=R.ingestion_run_dir(f"quick_extract_{source}"))
        finally:
            # Always clean up the temporary profile, even on failure
            mgr.delete_profile(profile.name)

        kept = report.total_after_dedup or 0

        return {
            "status": "completed" if kept > 0 else "no_frames",
            "dataset_name": profile.target_dataset,
            "source_videos": len(videos),
            "total_sampled": report.total_sampled,
            "after_quality": report.total_kept_after_quality,
            "after_dedup": report.total_after_dedup,
            "kept_frames": kept,
            "report_summary": report.summary(),
            "report": {
                "sources": report.total_sources, "total_raw_frames": report.total_raw_frames,
                "total_sampled": report.total_sampled, "kept_after_quality": report.total_kept_after_quality,
                "after_dedup": report.total_after_dedup, "removed_blurry": report.quality_removed_blurry,
                "removed_black": report.quality_removed_black, "removed_overexposed": report.quality_removed_overexposed,
                "removed_small": report.quality_removed_small, "dedup_removed": report.dedup_removed,
            },
        }
    except ImportError as e:
        raise HTTPException(500, f"Ingestion pipeline not available: {e}")
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.exception("Extraction failed with known error")
        raise HTTPException(400, f"Extraction failed: {e}")
    except Exception as e:
        logger.exception("Extraction failed with unexpected error")
        raise HTTPException(500, f"Extraction failed: {e}")
