"""AssetManager — 统一资产管理器。

v2.2: AssetManager 不管理图片存储，只管理 Manifest 引用。
storage/ 中的文件由用户直接放置，AssetManager 只负责记录引用路径。

职责：
  - 创建 Manifest（从文件路径列表生成 JSON Lines）
  - 保存 Manifest 到 dataset 版本目录
  - 解析 storage 路径为绝对路径
  - 将图片硬链接到 workspace
"""

from __future__ import annotations

import json
import os as _os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import R


class AssetManager:
    """统一资产管理器。

    不复制、不移动、不重命名任何文件。
    只管理 Manifest 中的引用路径。
    """

    def __init__(self):
        pass

    def create_manifest_record(
        self,
        storage_path: str,
        source: str = "",
        image_id: Optional[str] = None,
        width: int = 0,
        height: int = 0,
        sha256: str = "",
        capture_time: str = "",
        camera: str = "",
        **extra,
    ) -> Dict[str, Any]:
        """创建单条 Manifest 记录。

        Args:
            storage_path: 相对于项目根目录的路径，如 "storage/images/07-m01-01/xxx.jpeg"
            source: 来源标识
            image_id: 可选，自动生成
            width/height: 图片尺寸
        """
        if image_id is None:
            # 用文件名前缀作为 image_id
            stem = Path(storage_path).stem
            image_id = f"img_{stem[:20]}"

        record: Dict[str, Any] = {
            "image_id": image_id,
            "storage_path": storage_path,
        }

        # Auto-extract capture_time from filename if not provided
        if not capture_time:
            import re as _re
            stem = Path(storage_path).stem
            m = _re.search(r"(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})", stem)
            if m:
                y, mo, d, h, mi, s = m.groups()
                capture_time = f"{y}-{mo}-{d}T{h}:{mi}:{s}"

        # Auto-extract camera from filename if not provided
        if not camera:
            import re as _re
            m = _re.search(r"(\d+\.\d+\.\d+\.\d+)", storage_path)
            if m:
                camera = m.group(1)
        if source:
            record["source"] = source
        if width:
            record["width"] = width
        if height:
            record["height"] = height
        if sha256:
            record["sha256"] = sha256
        if capture_time:
            record["capture_time"] = capture_time
        if camera:
            record["camera"] = camera
        record.update(extra)
        return record

    def create_manifest(self, records: List[Dict]) -> List[Dict]:
        """从记录列表生成 Manifest（不做任何修改，直接返回）。"""
        return records

    def save_manifest(self, manifest: List[Dict], path: Path) -> None:
        """将 Manifest 写入 JSON Lines 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in manifest:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def resolve_path(self, storage_path: str) -> Path:
        """将 storage 路径（相对或绝对）解析为绝对路径。"""
        p = Path(storage_path)
        if p.is_absolute():
            return p
        # 相对于项目根目录
        return R.DATA.parent / p if not p.exists() else p.resolve()

    def open(self, path: str | Path) -> Any:
        p = self.resolve_path(str(path)) if isinstance(path, str) else path
        if p and p.exists():
            return p.open("rb")
        return None

    def copy(self, src: str | Path, dst: str | Path) -> Path:
        src_p = self.resolve_path(str(src)) if isinstance(src, str) else src
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_p), str(dst_p))
        return dst_p

    def compute_hash(self, path: str | Path, algorithm: str = "sha256") -> str:
        import hashlib as _hl
        p = self.resolve_path(str(path)) if isinstance(path, str) else path
        if not p or not p.exists():
            return ""
        h = _hl.new(algorithm)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def exists(self, path: str | Path) -> bool:
        p = self.resolve_path(str(path)) if isinstance(path, str) else path
        return p is not None and p.exists()

    def link_to_workspace(self, image_paths: List[Path], target_dir: Path) -> None:
        """将图片通过硬链接链接到 workspace 目录。

        不复制文件数据，空间占用几乎为 0。
        """
        images_dir = target_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for src in image_paths:
            if not src.exists():
                continue
            dst = images_dir / src.name
            if dst.exists():
                continue
            try:
                _os.link(str(src), str(dst))
            except (AttributeError, OSError):
                try:
                    dst.symlink_to(src)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
