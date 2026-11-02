"""raven-go 推理平台集成 —— 训练完成后自动注册 ONNX 模型。

P2（§7.11 训练入口）：训练服务与 Go 后端解耦部署，训练完成后通过 HTTP
将 ONNX 模型元数据注册到 raven-go 推理平台，实现「训练 → 部署」闭环。

设计要点：
  - 使用 stdlib urllib，零额外依赖（raven-trainer 未安装 httpx/requests）
  - raven-go API 地址通过环境变量 RAVEN_GO_API_URL 配置，默认 http://localhost:8080
  - 同机部署时直接传递 ONNX 文件路径；raven-go 侧从路径加载
  - 注册失败不影响训练流程（non-fatal），仅记录日志
  - 支持 multipart 文件上传（跨机部署场景），通过 upload=True 触发

注册端点契约（raven-go 侧需实现 POST /api/v1/models/register）：
  请求 JSON:
    {
      "model_key":   "person_intrusion_v1",
      "model_path":  "/abs/path/to/model.onnx",
      "task":        "detect|classify|seg|pose|obb",
      "project":     "person_intrusion",
      "quantization": {"half": false, "int8": false},
      "metrics":     {"mAP50": 0.85, ...}
    }
  响应 JSON:
    {"model_id": "...", "status": "registered"}
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def get_platform_url() -> str:
    """获取 raven-go 推理平台 API 地址。

    优先级：RAVEN_GO_API_URL 环境变量 > 默认 http://localhost:8080
    """
    return os.environ.get("RAVEN_GO_API_URL", "http://localhost:8080").rstrip("/")


def register_model_to_platform(
    onnx_path: str,
    model_name: str,
    task: str,
    project_name: str,
    metrics: Optional[Dict[str, Any]] = None,
    quantization: Optional[Dict[str, bool]] = None,
    model_key: Optional[str] = None,
    upload: bool = False,
    timeout: float = 15.0,
    log_func: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """将训练完成的 ONNX 模型注册到 raven-go 推理平台。

    Args:
        onnx_path: ONNX 模型文件绝对路径
        model_name: 模型名称（如 yolo11m）
        task: 任务类型（detect/classify/seg/pose/obb）
        project_name: 训练项目名
        metrics: 训练指标（mAP 等）
        quantization: 量化信息 {"half": bool, "int8": bool}
        model_key: 模型唯一标识，默认 "{project_name}_{model_name}"
        upload: 是否上传文件（跨机部署），False 则仅注册路径
        timeout: HTTP 请求超时（秒）
        log_func: 日志回调

    Returns:
        Dict with "success" (bool) and optional "model_id" / "error"
    """
    _log = log_func or (lambda msg: None)

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        return {"success": False, "error": f"ONNX file not found: {onnx_path}"}

    key = model_key or f"{project_name}_{model_name}"
    base_url = get_platform_url()

    payload: Dict[str, Any] = {
        "model_key": key,
        "model_name": model_name,
        "model_path": str(onnx_file.resolve()),
        "task": task,
        "project": project_name,
        "file_size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
        "quantization": quantization or {"half": False, "int8": False},
        "metrics": metrics or {},
    }

    url = f"{base_url}/api/v1/models/register"
    _log(f"[Platform] Registering model '{key}' to {url}")

    try:
        if upload:
            # 跨机部署：multipart 文件上传
            return _upload_and_register(url, onnx_file, payload, timeout, _log)
        else:
            # 同机部署：仅注册路径
            return _post_json(url, payload, timeout, _log)
    except urllib.error.URLError as e:
        _log(f"[Platform] Connection failed: {e}")
        return {"success": False, "error": f"connection_failed: {e}"}
    except Exception as e:
        _log(f"[Platform] Registration error: {e}")
        return {"success": False, "error": str(e)}


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    log_func: Callable[[str], None],
) -> Dict[str, Any]:
    """发送 JSON 注册请求。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        result = json.loads(body) if body else {}
        log_func(f"[Platform] Registration response: {resp.status}")
        return {
            "success": resp.status < 400,
            "model_id": result.get("model_id", ""),
            "status": result.get("status", "registered"),
            "detail": result,
        }


def _upload_and_register(
    url: str,
    onnx_file: Path,
    metadata: Dict[str, Any],
    timeout: float,
    log_func: Callable[[str], None],
) -> Dict[str, Any]:
    """multipart 文件上传注册（跨机部署）。

    构造 multipart/form-data：metadata（JSON） + file（ONNX 二进制）。
    """
    boundary = "----RavenTrainerBoundary7MA4YWxkTrZu7gR"
    metadata_json = json.dumps(metadata)

    body_parts: list[bytes] = []
    # metadata field
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        b'Content-Disposition: form-data; name="metadata"\r\n'
        b"Content-Type: application/json\r\n\r\n"
    )
    body_parts.append(metadata_json.encode() + b"\r\n")
    # file field
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        (
            f'Content-Disposition: form-data; name="file"; filename="{onnx_file.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
    )
    body_parts.append(onnx_file.read_bytes() + b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())

    data = b"".join(body_parts)
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        result = json.loads(body) if body else {}
        log_func(f"[Platform] Upload response: {resp.status}")
        return {
            "success": resp.status < 400,
            "model_id": result.get("model_id", ""),
            "status": result.get("status", "registered"),
            "detail": result,
        }
