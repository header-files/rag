"""Flat multi-instance endpoint configs (embedding, rerank, and later peers)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from common.config.env import load_env
from common.config.errors import ConfigError
from common.logger import get_logger

log = get_logger(__name__)

_INT_EXTRA_KEYS = {"dimensions"}


@dataclass
class EndpointConfig:
    id: str
    protocol: str
    name: str
    base_url: str
    api_key: str
    timeout_seconds: float = 60.0
    extra: dict[str, str] = field(default_factory=dict)


def extra_json_fields(extra: dict[str, str]) -> dict[str, object]:
    """Forward endpoint extras into a JSON body.

    ``dimensions`` is sent as an integer. Other extras are sent as strings.
    Unknown keys are not dropped; the provider reports them if unsupported.
    """
    body: dict[str, object] = {}
    for key, value in extra.items():
        if key in _INT_EXTRA_KEYS:
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ConfigError(
                    message=f"Invalid integer for {key}: {value}"
                ) from exc
            if parsed <= 0:
                raise ConfigError(message=f"{key} must be >= 1, got {parsed}")
            body[key] = parsed
        else:
            body[key] = value
    return body


def load_endpoints(
    *,
    key_prefix: str,
    default_key: str,
    valid_protocols: set[str],
    default_protocol: str,
    kind: str,
    env_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[str, dict[str, EndpointConfig]]:
    """Load ``{PREFIX}<ID>_*`` endpoints.

    ``key_prefix`` includes the trailing underscore, for example ``EMBEDDING_``.
    Returns ``(default_id, endpoints_by_id)``.
    """
    if not key_prefix.endswith("_"):
        raise ConfigError(message=f"key_prefix must end with '_': {key_prefix}")

    values = load_env(
        env_path,
        environ=environ,
        prefixes=(key_prefix,),
        keys=(default_key,),
    )
    pattern = re.compile(rf"^{re.escape(key_prefix)}([A-Z0-9]+)_(.+)$")
    grouped: dict[str, dict[str, str]] = {}
    for key, value in values.items():
        if value is None:
            continue
        match = pattern.match(key)
        if not match:
            continue
        endpoint_id = match.group(1).lower()
        field_name = match.group(2).upper()
        grouped.setdefault(endpoint_id, {})[field_name] = value.strip()

    if not grouped:
        raise ConfigError(
            message=(
                f"No {kind} endpoints found. Define "
                f"{key_prefix}<ID>_NAME / BASE_URL / API_KEY in .env"
            )
        )

    endpoints = {
        endpoint_id: _build_endpoint(
            endpoint_id,
            fields,
            key_prefix=key_prefix,
            valid_protocols=valid_protocols,
            default_protocol=default_protocol,
        )
        for endpoint_id, fields in grouped.items()
    }

    default_raw = values.get(default_key)
    if default_raw and str(default_raw).strip():
        default_id = str(default_raw).strip().lower()
    else:
        default_id = next(iter(endpoints))

    if default_id not in endpoints:
        raise ConfigError(message=f"{default_key} '{default_id}' is not defined")

    log.info(
        "Loaded %s configs default=%s count=%d ids=%s",
        kind,
        default_id,
        len(endpoints),
        ",".join(sorted(endpoints)),
    )
    return default_id, endpoints


def get_endpoint(
    endpoint_id: str | None,
    *,
    key_prefix: str,
    default_key: str,
    valid_protocols: set[str],
    default_protocol: str,
    kind: str,
    env_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> EndpointConfig:
    default_id, endpoints = load_endpoints(
        key_prefix=key_prefix,
        default_key=default_key,
        valid_protocols=valid_protocols,
        default_protocol=default_protocol,
        kind=kind,
        env_path=env_path,
        environ=environ,
    )
    target = endpoint_id.strip().lower() if endpoint_id else default_id
    if target not in endpoints:
        known = ", ".join(sorted(endpoints))
        raise ConfigError(message=f"Unknown {kind} '{target}'. Known: {known}")
    cfg = endpoints[target]
    log.info(
        "Selected %s id=%s protocol=%s name=%s base_url=%s",
        kind,
        cfg.id,
        cfg.protocol,
        cfg.name,
        cfg.base_url,
    )
    return cfg


def _build_endpoint(
    endpoint_id: str,
    fields: dict[str, str],
    *,
    key_prefix: str,
    valid_protocols: set[str],
    default_protocol: str,
) -> EndpointConfig:
    required = ("NAME", "BASE_URL", "API_KEY")
    missing = [name for name in required if not fields.get(name)]
    if missing:
        raise ConfigError(
            message=(
                f"{key_prefix}{endpoint_id.upper()} missing required fields: "
                + ", ".join(
                    f"{key_prefix}{endpoint_id.upper()}_{name}" for name in missing
                )
            )
        )

    protocol_raw = fields.get("PROTOCOL") or default_protocol
    protocol = protocol_raw.strip().lower()
    if protocol not in valid_protocols:
        raise ConfigError(
            message=(
                f"Unsupported protocol for {key_prefix}{endpoint_id.upper()}: "
                f"{protocol_raw}"
            )
        )

    timeout_raw = fields.get("TIMEOUT_SECONDS", "60")
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError(
            message=(
                f"{key_prefix}{endpoint_id.upper()} has invalid "
                f"TIMEOUT_SECONDS: {timeout_raw}"
            )
        ) from exc
    if timeout <= 0:
        raise ConfigError(
            message=(
                f"{key_prefix}{endpoint_id.upper()}_TIMEOUT_SECONDS "
                f"must be > 0, got {timeout}"
            )
        )

    known = {"PROTOCOL", "NAME", "BASE_URL", "API_KEY", "TIMEOUT_SECONDS"}
    extra = {key.lower(): value for key, value in fields.items() if key not in known}
    if "dimensions" in extra:
        extra_json_fields({"dimensions": extra["dimensions"]})

    return EndpointConfig(
        id=endpoint_id,
        protocol=protocol,
        name=fields["NAME"],
        base_url=fields["BASE_URL"].rstrip("/"),
        api_key=fields["API_KEY"],
        timeout_seconds=timeout,
        extra=extra,
    )
