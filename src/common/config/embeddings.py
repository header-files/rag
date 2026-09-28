"""Embedding endpoint settings (``EMBEDDING_<ID>_*``)."""

from __future__ import annotations

from pathlib import Path

from common.config.endpoints import EndpointConfig, get_endpoint, load_endpoints

_VALID_PROTOCOLS = {"openai-embeddings"}
_DEFAULT_PROTOCOL = "openai-embeddings"


def load_embeddings(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, dict[str, EndpointConfig]]:
    """Return ``(default_embedding_id, embeddings_by_id)``."""
    return load_endpoints(
        key_prefix="EMBEDDING_",
        default_key="DEFAULT_EMBEDDING",
        valid_protocols=_VALID_PROTOCOLS,
        default_protocol=_DEFAULT_PROTOCOL,
        kind="embedding",
        env_path=env_path,
        environ=environ,
    )


def get_embedding(
    embedding_id: str | None = None,
    *,
    env_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> EndpointConfig:
    return get_endpoint(
        embedding_id,
        key_prefix="EMBEDDING_",
        default_key="DEFAULT_EMBEDDING",
        valid_protocols=_VALID_PROTOCOLS,
        default_protocol=_DEFAULT_PROTOCOL,
        kind="embedding",
        env_path=env_path,
        environ=environ,
    )
