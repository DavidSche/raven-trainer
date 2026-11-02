"""Integration with external platforms (raven-go inference, etc.)."""

from .raven_go import register_model_to_platform, get_platform_url

__all__ = ["register_model_to_platform", "get_platform_url"]
