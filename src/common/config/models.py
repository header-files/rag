"""Model connection settings loaded from flat .env keys."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from common.config.env import load_env
from common.config.errors import ConfigError

ProtocolId = Literal["openai-completions", "openai-responses", "anthropic-messages"]

_MODEL_KEY_RE = re.compile(r"^MODEL_([A-Z0-9]+)_(.+)$")
_REQUIRED_FIELDS = ("PROTOCOL", "NAME", "BASE_URL", "API_KEY")
_DEFAULT_PROTOCOL: ProtocolId = "openai-completions"
_VALID_PROTOCOLS: set[str] = {
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
}


@dataclass
class ModelConfig:
    id: str
    protocol: ProtocolId
    name: str
    base_url: str
    api_key: str
    timeout_seconds: float = 60.0
    anthropic_version: str = "2023-06-01"
    extra: dict[str, str] = field(default_factory=dict)


def _normalize_model_id(raw: str) -> str:
    return raw.lower()


def _parse_protocol(value: str | None) -> ProtocolId:
    if value is None or value.strip() == "":
        return _DEFAULT_PROTOCOL
    normalized = value.strip().lower()
    if normalized not in _VALID_PROTOCOLS:
        raise ConfigError(message=f"Unsupported protocol: {value}")
    return normalized  # type: ignore[return-value]


def _group_model_vars(env: dict[str, str | None]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for key, value in env.items():
        if value is None:
            continue
        match = _MODEL_KEY_RE.match(key)
        if not match:
            continue
        model_id = _normalize_model_id(match.group(1))
        field_name = match.group(2).upper()
        grouped.setdefault(model_id, {})[field_name] = value.strip()
    return grouped


def _build_model_config(model_id: str, fields: dict[str, str]) -> ModelConfig:
    missing = [name for name in _REQUIRED_FIELDS if not fields.get(name)]
    if "PROTOCOL" in missing:
        missing.remove("PROTOCOL")
    if missing:
        raise ConfigError(
            message=(
                f"Model '{model_id}' missing required fields: "
                + ", ".join(f"MODEL_{model_id.upper()}_{f}" for f in missing)
            )
        )

    timeout_raw = fields.get("TIMEOUT_SECONDS", "60")
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError(
            message=f"Model '{model_id}' has invalid TIMEOUT_SECONDS: {timeout_raw}"
        ) from exc

    known = {
        "PROTOCOL",
        "NAME",
        "BASE_URL",
        "API_KEY",
        "TIMEOUT_SECONDS",
        "ANTHROPIC_VERSION",
    }
    extra = {k.lower(): v for k, v in fields.items() if k not in known}

    return ModelConfig(
        id=model_id,
        protocol=_parse_protocol(fields.get("PROTOCOL")),
        name=fields["NAME"],
        base_url=fields["BASE_URL"].rstrip("/"),
        api_key=fields["API_KEY"],
        timeout_seconds=timeout,
        anthropic_version=fields.get("ANTHROPIC_VERSION", "2023-06-01"),
        extra=extra,
    )


def load_models(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, dict[str, ModelConfig]]:
    """Load models from .env (+ optional process env).

    Returns (default_model_id, models_by_id).
    """
    values = load_env(
        env_path,
        environ=environ,
        prefixes=("MODEL_",),
        keys=("DEFAULT_MODEL",),
    )

    grouped = _group_model_vars(values)
    if not grouped:
        raise ConfigError(
            message=(
                "No models found. Define MODEL_<ID>_NAME / BASE_URL / API_KEY in .env"
            )
        )

    models = {
        model_id: _build_model_config(model_id, fields)
        for model_id, fields in grouped.items()
    }

    default_raw = values.get("DEFAULT_MODEL")
    if default_raw and default_raw.strip():
        default_id = _normalize_model_id(default_raw.strip())
    else:
        default_id = next(iter(models))

    if default_id not in models:
        raise ConfigError(message=f"DEFAULT_MODEL '{default_id}' is not defined")

    return default_id, models


def get_model(
    model_id: str | None = None,
    *,
    env_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> ModelConfig:
    default_id, models = load_models(env_path, environ=environ)
    target = _normalize_model_id(model_id) if model_id else default_id
    if target not in models:
        known = ", ".join(sorted(models))
        raise ConfigError(message=f"Unknown model '{target}'. Known: {known}")
    return models[target]
