"""raven-train ui — 启动 Web UI。

Usage:
    raven-train ui                       # 默认 127.0.0.1:8081
    raven-train ui --port 3000           # 指定端口
    raven-train ui --open                # 自动打开浏览器
"""

from __future__ import annotations

import click


@click.command(name="ui")
@click.option("--port", type=int, default=8081, help="Server port (default: 8081)")
@click.option("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
@click.option("--open", "open_browser", is_flag=True, help="Open browser automatically")
def cmd_ui(port: int, host: str, open_browser: bool):
    """启动 Raven Trainer Web UI 服务。"""
    from ..api.server import run_server
    run_server(port=port, host=host, open_browser=open_browser)
