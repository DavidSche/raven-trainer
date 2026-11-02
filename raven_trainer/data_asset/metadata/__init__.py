"""Metadata — 每张图片的结构化元数据。

不仅仅是 filename + label，而是完整的结构化信息，支持查询和过滤。

每一张图片都有：
  - 来源信息：camera, video, frame, timestamp
  - 设备信息：workshop, device, operator, shift
  - 场景信息：scene, lighting, weather
  - 图像信息：resolution, file_size, hash
  - 质量信息：blur_score, brightness, quality_score（与 Quality 模块共享）

M-25: 主存储改为 SQLite 索引，JSONL 保留作为人类可读备份。
query() 使用 SQL WHERE 子句，add() 双写 SQLite + JSONL。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS image_metadata (
    image_id      TEXT PRIMARY KEY,
    filename      TEXT,
    storage_path  TEXT,
    source        TEXT,
    camera        TEXT,
    video         TEXT,
    frame_index   INTEGER,
    timestamp     TEXT,
    workshop      TEXT,
    device        TEXT,
    operator      TEXT,
    shift         TEXT,
    scene         TEXT,
    lighting      TEXT,
    weather       TEXT,
    width         INTEGER,
    height        INTEGER,
    file_size     INTEGER,
    file_hash     TEXT,
    blur_score    REAL,
    brightness    REAL,
    quality_score REAL,
    label_count   INTEGER,
    has_annotation INTEGER
);

CREATE INDEX IF NOT EXISTS idx_camera      ON image_metadata(camera);
CREATE INDEX IF NOT EXISTS idx_scene       ON image_metadata(scene);
CREATE INDEX IF NOT EXISTS idx_workshop    ON image_metadata(workshop);
CREATE INDEX IF NOT EXISTS idx_source      ON image_metadata(source);
CREATE INDEX IF NOT EXISTS idx_quality     ON image_metadata(quality_score);
CREATE INDEX IF NOT EXISTS idx_storage_path ON image_metadata(storage_path);
"""

# Columns in the same order as the INSERT statement
_DB_COLUMNS = [
    "image_id", "filename", "storage_path", "source",
    "camera", "video", "frame_index", "timestamp",
    "workshop", "device", "operator", "shift",
    "scene", "lighting", "weather",
    "width", "height", "file_size", "file_hash",
    "blur_score", "brightness", "quality_score",
    "label_count", "has_annotation",
]

# ``query()`` interpolates column names into SQL, so filters are whitelisted.
_COLUMNS = frozenset(_DB_COLUMNS)

_INSERT_SQL = f"""
INSERT OR REPLACE INTO image_metadata (
    {', '.join(_DB_COLUMNS)}
) VALUES ({', '.join('?' for _ in _DB_COLUMNS)})
"""


@dataclass
class ImageMetadata:
    """单张图片的完整结构化元数据。"""

    # 标识
    image_id: str                            # 唯一标识符
    filename: str                            # 文件名
    storage_path: str = ""                   # 存储路径（相对于项目根目录）
    source: str = ""                         # 来源名称

    # 来源
    camera: str = ""                         # 摄像头编号
    video: str = ""                          # 源视频文件
    frame_index: int = 0                     # 视频中的帧号
    timestamp: str = ""                      # 时间戳 ISO 8601

    # 设备环境
    workshop: str = ""                       # 车间
    device: str = ""                         # 设备
    operator: str = ""                       # 操作员
    shift: str = ""                          # 班次：day / night / dawn / dusk

    # 场景
    scene: str = ""                          # 场景标签
    lighting: str = ""                       # 光照：bright / normal / dark / backlight
    weather: str = ""                        # 天气：clear / fog / rain

    # 图像属性
    width: int = 0
    height: int = 0
    file_size: int = 0
    file_hash: str = ""                      # SHA256

    # 质量（与 Quality 模块共享）
    blur_score: float = 0.0                  # Laplacian 方差
    brightness: float = 0.0                  # 平均亮度
    quality_score: float = 0.0               # 0-100 综合质量分

    # 标签
    tags: List[str] = field(default_factory=list)

    # 标注
    label_count: int = 0
    has_annotation: bool = False

    # 扩展
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "storage_path": self.storage_path,
            "source": self.source,
            "camera": self.camera,
            "video": self.video,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "workshop": self.workshop,
            "device": self.device,
            "operator": self.operator,
            "shift": self.shift,
            "scene": self.scene,
            "lighting": self.lighting,
            "weather": self.weather,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "quality_score": self.quality_score,
            "tags": self.tags,
            "label_count": self.label_count,
            "has_annotation": self.has_annotation,
            "extra": self.extra,
        }


class MetadataStore:
    def __init__(self, metadata_dir):
        # Coerce str -> Path so callers can pass either
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        # JSONL paths (kept for backward-compatible human-readable backup)
        self._image_path = self.metadata_dir / "image.jsonl"
        self._quality_path = self.metadata_dir / "quality.jsonl"
        self._source_path = self.metadata_dir / "source.jsonl"
        self._legacy_path = self.metadata_dir / "metadata.jsonl"
        # SQLite (M-25: primary query engine)
        self._db_path = self.metadata_dir / "metadata.db"
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Open (or create) the SQLite database and ensure schema exists."""
        self._db = sqlite3.connect(str(self._db_path))
        self._db.executescript(_SCHEMA_SQL)
        self._db.commit()

    def _require_db(self) -> sqlite3.Connection:
        """Return the live connection or fail loudly.

        Replaces bare ``assert self._db is not None``: assertions disappear
        under ``python -O``, which would turn every method into an
        ``AttributeError: 'NoneType' object has no attribute 'execute'``.
        """
        if self._db is None:
            raise RuntimeError("MetadataStore is closed")
        return self._db

    def _db_params(self, meta: ImageMetadata) -> tuple:
        """Extract values matching _DB_COLUMNS order for SQL parameters."""
        return (
            meta.image_id, meta.filename, meta.storage_path, meta.source,
            meta.camera, meta.video, meta.frame_index, meta.timestamp,
            meta.workshop, meta.device, meta.operator, meta.shift,
            meta.scene, meta.lighting, meta.weather,
            meta.width, meta.height, meta.file_size, meta.file_hash,
            meta.blur_score, meta.brightness, meta.quality_score,
            meta.label_count, int(meta.has_annotation),
        )

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> ImageMetadata:
        """Convert a SQLite row to an ImageMetadata instance."""
        d = dict(row)
        # Restore bool from int
        if "has_annotation" in d:
            d["has_annotation"] = bool(d["has_annotation"])
        # Fill defaults for fields not in the table (tags, extra, operator, shift, weather)
        d.setdefault("tags", [])
        d.setdefault("extra", {})
        d.setdefault("operator", "")
        d.setdefault("shift", "")
        d.setdefault("weather", "")
        return ImageMetadata(**d)

    # ------------------------------------------------------------------
    # JSONL (backward-compatible backup)
    # ------------------------------------------------------------------

    def _write_jsonl(self, meta: ImageMetadata) -> None:
        """Append records to the 3 JSONL files (human-readable backup)."""
        image_rec = {"image_id": meta.image_id, "filename": meta.filename,
                     "width": meta.width, "height": meta.height}
        quality_rec = {"image_id": meta.image_id, "blur_score": meta.blur_score,
                       "brightness": meta.brightness, "quality_score": meta.quality_score}
        source_rec = {"image_id": meta.image_id, "camera": meta.camera,
                      "frame_index": meta.frame_index, "timestamp": meta.timestamp}
        with open(self._image_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(image_rec, ensure_ascii=False) + "\n")
        with open(self._quality_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(quality_rec, ensure_ascii=False) + "\n")
        with open(self._source_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(source_rec, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path):
        """Read a JSONL file into a list of dicts (used for migration/fallback)."""
        if not path.exists():
            return []
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, meta: ImageMetadata) -> None:
        """Add a metadata record. Writes to both SQLite and JSONL."""
        # 1. SQLite (primary)
        db = self._require_db()
        db.execute(_INSERT_SQL, self._db_params(meta))
        db.commit()
        # 2. JSONL (backup)
        self._write_jsonl(meta)

    def add_batch(self, metas: List[ImageMetadata]) -> None:
        """Add multiple records efficiently (single transaction for SQLite)."""
        db = self._require_db()
        with db:
            for meta in metas:
                db.execute(_INSERT_SQL, self._db_params(meta))
        # JSONL append
        for meta in metas:
            self._write_jsonl(meta)

    def query(self, **filters) -> List[ImageMetadata]:
        """Query metadata using SQL WHERE clauses (M-25: was O(3N) full load).

        Column names are whitelisted: they are interpolated into the SQL string,
        so an unknown/attacker-supplied key must never reach ``execute``.
        """
        db = self._require_db()
        db.row_factory = sqlite3.Row
        unknown = set(filters) - _COLUMNS
        if unknown:
            raise ValueError(f"Unknown metadata column(s): {sorted(unknown)}")
        if not filters:
            rows = db.execute(
                "SELECT * FROM image_metadata LIMIT 10000"
            ).fetchall()
        else:
            clauses = []
            params: list = []
            for key, value in filters.items():
                clauses.append(f"{key} = ?")
                params.append(int(value) if key == "has_annotation" else value)
            where = " AND ".join(clauses)
            rows = db.execute(
                f"SELECT * FROM image_metadata WHERE {where}", params
            ).fetchall()
        return [self._row_to_metadata(r) for r in rows]

    def count(self) -> int:
        """Return total image count."""
        db = self._require_db()
        row = db.execute("SELECT COUNT(*) FROM image_metadata").fetchone()
        return row[0] if row else 0

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        db = self._require_db()
        total = self.count()
        if total == 0:
            return {}
        # Cameras
        cam_rows = db.execute(
            "SELECT DISTINCT camera FROM image_metadata WHERE camera != ''"
        ).fetchall()
        cameras = [r[0] for r in cam_rows]
        # Avg quality
        row = db.execute(
            "SELECT AVG(quality_score) FROM image_metadata"
        ).fetchone()
        avg_q = row[0] if row and row[0] is not None else 0.0
        return {
            "total_images": total,
            "cameras": cameras,
            "avg_quality_score": avg_q,
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._db:
            self._db.close()
            self._db = None

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        # Last-resort cleanup: SQLite connections hold a file handle that
        # prevents the metadata dir from being deleted/moved on Windows.
        try:
            self.close()
        except Exception:
            pass

    def migrate_from_jsonl(self) -> int:
        """One-time migration: import existing JSONL data into SQLite.

        Returns the number of records migrated.
        """
        images = {r["image_id"]: r for r in self._read_jsonl(self._image_path)}
        qualities = {r["image_id"]: r for r in self._read_jsonl(self._quality_path)}
        sources = {r["image_id"]: r for r in self._read_jsonl(self._source_path)}
        all_ids = set(images.keys()) | set(qualities.keys()) | set(sources.keys())
        if not all_ids:
            return 0
        count = 0
        db = self._require_db()
        with db:
            for img_id in all_ids:
                merged: Dict[str, Any] = {}
                merged.update(images.get(img_id, {}))
                merged.update(qualities.get(img_id, {}))
                merged.update(sources.get(img_id, {}))
                # Provide defaults for fields not in JSONL
                merged.setdefault("storage_path", "")
                merged.setdefault("source", "")
                merged.setdefault("camera", "")
                merged.setdefault("video", "")
                merged.setdefault("frame_index", 0)
                merged.setdefault("timestamp", "")
                merged.setdefault("workshop", "")
                merged.setdefault("device", "")
                merged.setdefault("operator", "")
                merged.setdefault("shift", "")
                merged.setdefault("scene", "")
                merged.setdefault("lighting", "")
                merged.setdefault("weather", "")
                merged.setdefault("width", 0)
                merged.setdefault("height", 0)
                merged.setdefault("file_size", 0)
                merged.setdefault("file_hash", "")
                merged.setdefault("blur_score", 0.0)
                merged.setdefault("brightness", 0.0)
                merged.setdefault("quality_score", 0.0)
                merged.setdefault("label_count", 0)
                merged.setdefault("has_annotation", 0)
                meta = ImageMetadata(**merged)
                db.execute(_INSERT_SQL, self._db_params(meta))
                count += 1
        return count
