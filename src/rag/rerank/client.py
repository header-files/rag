"""Rerank client."""

from __future__ import annotations

from pathlib import Path

from common.config.endpoints import EndpointConfig
from common.config.rerank import get_reranker
from common.logger import get_logger
from llm.transport.http import HttpTransport
from rag.rerank.base import RerankResult
from rag.rerank.registry import create_rerank_adapter

log = get_logger(__name__)


class RerankClient:
    def __init__(
        self,
        config: EndpointConfig,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        self._adapter = create_rerank_adapter(config, transport=transport)

    @classmethod
    def from_env(
        cls,
        rerank_id: str | None = None,
        *,
        env_path: str | Path | None = None,
    ) -> RerankClient:
        config = get_reranker(rerank_id, env_path=env_path)
        return cls(config)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []
        effective_top_n = top_n
        if effective_top_n is not None:
            effective_top_n = min(effective_top_n, len(documents))
        log.info(
            "rerank start model_id=%s protocol=%s name=%s documents=%d top_n=%s",
            self.config.id,
            self.config.protocol,
            self.config.name,
            len(documents),
            effective_top_n,
        )
        try:
            results = self._adapter.rerank(
                query,
                documents,
                top_n=effective_top_n,
            )
        except Exception as exc:
            log.error(
                "rerank failed model_id=%s protocol=%s error=%s",
                self.config.id,
                self.config.protocol,
                exc,
            )
            raise
        log.info("rerank done model_id=%s results=%d", self.config.id, len(results))
        return results
