"""Public HTTP API listen / client address settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config.env import load_env
from common.config.parsing import parse_float, parse_int
from common.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 120.0


def load_api_settings(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> ApiSettings:
    values = load_env(env_path, environ=environ, prefixes=("API_",))
    host = _optional_str(values.get("API_HOST")) or "127.0.0.1"
    port = parse_int(
        _as_str(values.get("API_PORT")),
        default=8000,
        field="API_PORT",
        minimum=1,
        maximum=65535,
    )
    base = _optional_str(values.get("API_BASE_URL")) or f"http://{host}:{port}"
    timeout = parse_float(
        _as_str(values.get("API_TIMEOUT_SECONDS")),
        default=120.0,
        field="API_TIMEOUT_SECONDS",
        minimum=0.1,
    )
    settings = ApiSettings(
        host=host,
        port=port,
        base_url=base.rstrip("/"),
        timeout_seconds=timeout,
    )
    log.info(
        "Loaded api settings host=%s port=%d base_url=%s",
        settings.host,
        settings.port,
        settings.base_url,
    )
    return settings


def _as_str(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
