"""Small parsers shared by config domains."""

from __future__ import annotations

from common.config.errors import ConfigError

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_bool(raw: str | None, *, default: bool, field: str) -> bool:
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(message=f"Invalid boolean for {field}: {raw}")


def parse_int(
    raw: str | None,
    *,
    default: int,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ConfigError(message=f"Invalid integer for {field}: {raw}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(message=f"{field} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(message=f"{field} must be <= {maximum}, got {value}")
    return value


def parse_float(
    raw: str | None,
    *,
    default: float,
    field: str,
    minimum: float | None = None,
) -> float:
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise ConfigError(message=f"Invalid number for {field}: {raw}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(message=f"{field} must be >= {minimum}, got {value}")
    return value
