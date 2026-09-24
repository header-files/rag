"""Logging settings loaded from .env."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config.env import load_env
from common.config.errors import ConfigError

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(raw: str | None, *, default: bool, field: str) -> bool:
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(message=f"Invalid boolean for {field}: {raw}")


def _parse_int(raw: str | None, *, default: int, field: str) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(message=f"Invalid integer for {field}: {raw}") from exc
    if value < 0:
        raise ConfigError(message=f"{field} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class LogSettings:
    level: str = "INFO"
    to_console: bool = True
    to_file: bool = False
    file_path: Path = Path("logs/app.log")
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    fmt: str = (
        "%(asctime)s | %(levelname)s | %(trace_id)s | %(name)s | %(message)s"
    )
    ensure_fields: tuple[str, ...] = ("trace_id",)


def load_log_settings(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> LogSettings:
    """Load logging settings from LOG_* keys in .env / process env."""
    values = load_env(
        env_path,
        environ=environ,
        prefixes=("LOG_",),
    )

    level = (values.get("LOG_LEVEL") or "INFO").strip().upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if level not in valid_levels:
        raise ConfigError(message=f"Invalid LOG_LEVEL: {level}")

    file_raw = values.get("LOG_FILE_PATH") or "logs/app.log"
    max_bytes = _parse_int(
        values.get("LOG_MAX_BYTES"),
        default=10 * 1024 * 1024,
        field="LOG_MAX_BYTES",
    )
    backup_count = _parse_int(
        values.get("LOG_BACKUP_COUNT"),
        default=5,
        field="LOG_BACKUP_COUNT",
    )
    to_file = _parse_bool(
        values.get("LOG_TO_FILE"), default=False, field="LOG_TO_FILE"
    )
    if to_file and max_bytes <= 0:
        raise ConfigError(message="LOG_MAX_BYTES must be > 0 when LOG_TO_FILE=true")

    fmt = values.get("LOG_FORMAT") or (
        "%(asctime)s | %(levelname)s | %(trace_id)s | %(name)s | %(message)s"
    )

    ensure_raw = values.get("LOG_ENSURE_FIELDS") or "trace_id"
    ensure_fields = tuple(
        part.strip() for part in ensure_raw.split(",") if part.strip()
    )
    if not ensure_fields:
        raise ConfigError(message="LOG_ENSURE_FIELDS must list at least one field")

    return LogSettings(
        level=level,
        to_console=_parse_bool(
            values.get("LOG_TO_CONSOLE"), default=True, field="LOG_TO_CONSOLE"
        ),
        to_file=to_file,
        file_path=Path(file_raw.strip()),
        max_bytes=max_bytes if max_bytes > 0 else 10 * 1024 * 1024,
        backup_count=backup_count,
        fmt=fmt.strip(),
        ensure_fields=ensure_fields,
    )
