"""Embedding adapter registry."""

from __future__ import annotations

from common.config.endpoints import EndpointConfig
from common.config.errors import ConfigError
from llm.transport.http import HttpTransport
from rag.embedding.base import EmbeddingAdapter
from rag.embedding.openai import OpenAIEmbeddingsAdapter

_REGISTRY: dict[str, type[EmbeddingAdapter]] = {
    "openai-embeddings": OpenAIEmbeddingsAdapter,
}


def create_embedding_adapter(
    config: EndpointConfig,
    transport: HttpTransport | None = None,
) -> EmbeddingAdapter:
    cls = _REGISTRY.get(config.protocol)
    if cls is None:
        raise ConfigError(
            message=(
                f"No embedding adapter registered for protocol '{config.protocol}'"
            )
        )
    return cls(config, transport=transport)
