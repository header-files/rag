"""Rerank endpoint settings (``RERANK_<ID>_*``)."""

from __future__ import annotations

from pathlib import Path

from common.config.endpoints import EndpointConfig, get_endpoint, load_endpoints

_VALID_PROTOCOLS = {"cohere-rerank", "dashscope-text-rerank"}
_DEFAULT_PROTOCOL = "cohere-rerank"


def load_rerankers(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, dict[str, EndpointConfig]]:
    """Return ``(default_rerank_id, rerankers_by_id)``."""
    return load_endpoints(
        key_prefix="RERANK_",
        default_key="DEFAULT_RERANK",
        valid_protocols=_VALID_PROTOCOLS,
        default_protocol=_DEFAULT_PROTOCOL,
        kind="rerank",
        env_path=env_path,
        environ=environ,
    )


def get_reranker(
    rerank_id: str | None = None,
    *,
    env_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> EndpointConfig:
    return get_endpoint(
        rerank_id,
        key_prefix="RERANK_",
        default_key="DEFAULT_RERANK",
        valid_protocols=_VALID_PROTOCOLS,
        default_protocol=_DEFAULT_PROTOCOL,
        kind="rerank",
        env_path=env_path,
        environ=environ,
    )
