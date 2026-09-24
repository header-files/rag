"""Logging setup: console and size-based rotating files."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from common.config.log_settings import LogSettings, load_log_settings
from common.logger.filters import ContextFilter

_CONFIGURED = False
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(trace_id)s | %(name)s | %(message)s"
)


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
      backup files (``app.log.1``, …) and continues writing to a new ``app.log``.
    - ContextFilter injects ``trace_id`` (and other bound context fields).

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
    context_filter = ContextFilter(ensure_fields=resolved.ensure_fields)

    if resolved.to_console:
        console = logging.StreamHandler()
        console.setLevel(resolved.level)
        console.setFormatter(formatter)
        console.addFilter(context_filter)
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
        file_handler.addFilter(ContextFilter(ensure_fields=resolved.ensure_fields))
        root.addHandler(file_handler)

    if not root.handlers:
        null = logging.NullHandler()
        null.addFilter(ContextFilter(ensure_fields=resolved.ensure_fields))
        root.addHandler(null)

    _CONFIGURED = True
    return resolved


def reset_logging_state() -> None:
    """Test helper: clear configuration flag, root handlers, and log context."""
    global _CONFIGURED
    from common.logger.context import clear

    root = logging.getLogger()
    root.handlers.clear()
    clear()
    _CONFIGURED = False
