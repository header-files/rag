"""Tests for logging settings, rotation, and trace context."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from common.config.errors import ConfigError
from common.config.log_settings import LogSettings, load_log_settings
from common.logger import (
    bind,
    get_logger,
    get_trace_id,
    log_context,
    new_trace_id,
    reset_logging_state,
    setup_logging,
    traced,
)


def test_load_log_settings_defaults() -> None:
    settings = load_log_settings(environ={})
    assert settings.to_console is True
    assert settings.to_file is False
    assert settings.level == "INFO"
    assert settings.max_bytes == 10 * 1024 * 1024
    assert settings.ensure_fields == ("trace_id",)
    assert "%(trace_id)s" in settings.fmt


def test_load_log_settings_from_env() -> None:
    settings = load_log_settings(
        environ={
            "LOG_LEVEL": "debug",
            "LOG_TO_CONSOLE": "false",
            "LOG_TO_FILE": "true",
            "LOG_FILE_PATH": "tmp/x.log",
            "LOG_MAX_BYTES": "1024",
            "LOG_BACKUP_COUNT": "3",
            "LOG_ENSURE_FIELDS": "trace_id,user_id",
        }
    )
    assert settings.level == "DEBUG"
    assert settings.to_console is False
    assert settings.to_file is True
    assert settings.file_path == Path("tmp/x.log")
    assert settings.max_bytes == 1024
    assert settings.backup_count == 3
    assert settings.ensure_fields == ("trace_id", "user_id")


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


def test_trace_id_appears_in_log_output(tmp_path: Path) -> None:
    reset_logging_state()
    log_file = tmp_path / "trace.log"
    setup_logging(
        LogSettings(
            level="INFO",
            to_console=False,
            to_file=True,
            file_path=log_file,
            max_bytes=10_000,
            backup_count=1,
        ),
        force=True,
    )
    logger = get_logger("test.trace")
    with traced("abc123trace"):
        assert get_trace_id() == "abc123trace"
        logger.info("traced-message")
    for h in logging.getLogger().handlers:
        h.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "abc123trace" in text
    assert "traced-message" in text
    assert get_trace_id() is None
    reset_logging_state()


def test_missing_trace_id_uses_placeholder(tmp_path: Path) -> None:
    reset_logging_state()
    log_file = tmp_path / "dash.log"
    setup_logging(
        LogSettings(
            level="INFO",
            to_console=False,
            to_file=True,
            file_path=log_file,
            max_bytes=10_000,
            backup_count=1,
        ),
        force=True,
    )
    get_logger("test.dash").info("no-trace")
    for h in logging.getLogger().handlers:
        h.flush()
    line = log_file.read_text(encoding="utf-8")
    assert " | - | " in line or line.split("|")[2].strip() == "-"
    reset_logging_state()


def test_bind_extra_field_for_future_extension(tmp_path: Path) -> None:
    reset_logging_state()
    log_file = tmp_path / "extra.log"
    setup_logging(
        LogSettings(
            level="INFO",
            to_console=False,
            to_file=True,
            file_path=log_file,
            max_bytes=10_000,
            backup_count=1,
            fmt="%(trace_id)s|%(user_id)s|%(message)s",
            ensure_fields=("trace_id", "user_id"),
        ),
        force=True,
    )
    logger = get_logger("test.extra")
    with log_context(trace_id="t1", user_id="u9"):
        logger.info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "t1|u9|hello" in log_file.read_text(encoding="utf-8")

    # unbound ensure field falls back to "-"
    open(log_file, "w").close()
    logger.info("again")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "-|-|again" in log_file.read_text(encoding="utf-8")
    reset_logging_state()


def test_new_trace_id_unique() -> None:
    assert new_trace_id() != new_trace_id()
    assert len(new_trace_id()) == 32


def test_bind_reset_isolation() -> None:
    reset_logging_state()
    token = bind(trace_id="outer", span_id="s1")
    assert get_trace_id() == "outer"
    with log_context(trace_id="inner"):
        assert get_trace_id() == "inner"
    assert get_trace_id() == "outer"
    from common.logger import reset

    reset(token)
    assert get_trace_id() is None
