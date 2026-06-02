"""
Centralized logging configuration for RAG Enterprise Pro.

Usage in any module:
    from logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("📦 Đã load thành công Index...")
    logger.warning("⚠️ Không tìm thấy kết quả...")
    logger.error("❌ Lỗi: ...")
    logger.debug("🔎 Queries: ...")

Log level can be controlled via env variable LOG_LEVEL (default: INFO):
    LOG_LEVEL=DEBUG python api.py
"""

import logging
import os
import sys

_LOG_CONFIGURED = False


def _default_format() -> str:
    """Use level-dependent format — errors get level prefix, info stays clean."""
    return "%(message)s"


def setup_logging(level: int | None = None) -> None:
    """Configure root logger once.

    Args:
        level: Log level (e.g. logging.DEBUG, logging.INFO).
               Falls back to env LOG_LEVEL or INFO.
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        level = getattr(logging, env_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers (e.g. from library imports)
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=_default_format()))

    root.addHandler(handler)
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger with centralised config applied.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
