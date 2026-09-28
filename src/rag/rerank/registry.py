"""Rerank adapter registry."""

from __future__ import annotations

from common.config.endpoints import EndpointConfig
from common.config.errors import ConfigError
from llm.transport.http import HttpTransport
from rag.rerank.base import RerankAdapter
from rag.rerank.cohere import CohereRerankAdapter
from rag.rerank.dashscope import DashScopeTextRerankAdapter

_REGISTRY: dict[str, type[RerankAdapter]] = {
    "cohere-rerank": CohereRerankAdapter,
    "dashscope-text-rerank": DashScopeTextRerankAdapter,
}


def create_rerank_adapter(
    config: EndpointConfig,
    transport: HttpTransport | None = None,
) -> RerankAdapter:
    cls = _REGISTRY.get(config.protocol)
    if cls is None:
        raise ConfigError(
            message=f"No rerank adapter registered for protocol '{config.protocol}'"
        )
    return cls(config, transport=transport)
