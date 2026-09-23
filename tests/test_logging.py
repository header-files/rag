"""Tests for logging settings and rotating file handler."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from common.config.errors import ConfigError
from common.config.log_settings import LogSettings, load_log_settings
from common.logger import get_logger, reset_logging_state, setup_logging


def test_load_log_settings_defaults() -> None:
    settings = load_log_settings(environ={})
    assert settings.to_console is True
    assert settings.to_file is False
    assert settings.level == "INFO"
    assert settings.max_bytes == 10 * 1024 * 1024


def test_load_log_settings_from_env() -> None:
    settings = load_log_settings(
        environ={
            "LOG_LEVEL": "debug",
            "LOG_TO_CONSOLE": "false",
            "LOG_TO_FILE": "true",
            "LOG_FILE_PATH": "tmp/x.log",
            "LOG_MAX_BYTES": "1024",
            "LOG_BACKUP_COUNT": "3",
        }
    )
    assert settings.level == "DEBUG"
    assert settings.to_console is False
    assert settings.to_file is True
    assert settings.file_path == Path("tmp/x.log")
    assert settings.max_bytes == 1024
    assert settings.backup_count == 3


def test_invalid_log_level() -> None:
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_log_settings(environ={"LOG_LEVEL": "verbose"})


def test_setup_console_and_file(tmp_path: Path) -> None:
    reset_logging_state()
    log_file = tmp_path / "app.log"
    setup_logging(
        LogSettings(
            level="INFO",
            to_console=True,
            to_file=True,
            file_path=log_file,
            max_bytes=1024,
            backup_count=2,
        ),
        force=True,
    )
    root = logging.getLogger()
    handler_types = {type(h).__name__ for h in root.handlers}
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types

    logger = get_logger("test.logger")
    logger.info("hello-file")
    for h in root.handlers:
        h.flush()
    assert "hello-file" in log_file.read_text(encoding="utf-8")
    reset_logging_state()


def test_rotating_file_creates_backup_instead_of_overwrite(tmp_path: Path) -> None:
    reset_logging_state()
    log_file = tmp_path / "rot.log"
    setup_logging(
        LogSettings(
            level="INFO",
            to_console=False,
            to_file=True,
            file_path=log_file,
            max_bytes=200,
            backup_count=3,
        ),
        force=True,
    )
    logger = get_logger("test.rotate")
    for i in range(50):
        logger.info("x" * 40 + f"-{i}")
    for h in logging.getLogger().handlers:
        h.flush()
        h.close()

    assert log_file.exists()
    assert (tmp_path / "rot.log.1").exists()
    reset_logging_state()


def test_both_disabled_uses_null_handler() -> None:
    reset_logging_state()
    setup_logging(
        LogSettings(to_console=False, to_file=False),
        force=True,
    )
    types = {type(h).__name__ for h in logging.getLogger().handlers}
    assert types == {"NullHandler"}
    reset_logging_state()
