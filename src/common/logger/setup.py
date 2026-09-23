"""Logging setup: console and size-based rotating files."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from common.config.log_settings import LogSettings, load_log_settings

_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger (call setup_logging first in app entrypoints)."""
    return logging.getLogger(name)


def setup_logging(
    settings: LogSettings | None = None,
    *,
    env_path: str | Path | None = None,
    force: bool = False,
) -> LogSettings:
    """Configure root logging handlers.

    - Console: StreamHandler (stderr)
    - File: RotatingFileHandler — when size exceeds ``max_bytes``, rotates to
      backup files (``app.log.1``, …) and continues writing to a new ``app.log``
      instead of truncating/overwriting in place without archival.

    Idempotent unless ``force=True``.
    """
    global _CONFIGURED
    resolved = settings or load_log_settings(env_path)

    if _CONFIGURED and not force:
        return resolved

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved.level)

    formatter = logging.Formatter(resolved.fmt)

    if resolved.to_console:
        console = logging.StreamHandler()
        console.setLevel(resolved.level)
        console.setFormatter(formatter)
        root.addHandler(console)

    if resolved.to_file:
        path = resolved.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=path,
            maxBytes=resolved.max_bytes,
            backupCount=resolved.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved.level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if not root.handlers:
        root.addHandler(logging.NullHandler())

    _CONFIGURED = True
    return resolved


def reset_logging_state() -> None:
    """Test helper: clear configuration flag and root handlers."""
    global _CONFIGURED
    root = logging.getLogger()
    root.handlers.clear()
    _CONFIGURED = False
