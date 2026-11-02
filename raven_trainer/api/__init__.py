"""Raven Trainer Web API.

FastAPI backend that wraps raven_trainer core modules.
Started via `raven-train ui`.
"""

from .server import app

__all__ = ["app"]
