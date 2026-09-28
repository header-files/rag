"""Embedding client."""

from __future__ import annotations

from pathlib import Path

from common.config.embeddings import get_embedding
from common.config.endpoints import EndpointConfig
from common.logger import get_logger
from llm.transport.http import HttpTransport
from rag.embedding.registry import create_embedding_adapter

log = get_logger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        config: EndpointConfig,
        *,
        batch_size: int = 16,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self._adapter = create_embedding_adapter(config, transport=transport)

    @classmethod
    def from_env(
        cls,
        embedding_id: str | None = None,
        *,
        env_path: str | Path | None = None,
        batch_size: int = 16,
    ) -> EmbeddingClient:
        config = get_embedding(embedding_id, env_path=env_path)
        return cls(config, batch_size=batch_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        log.info(
            "embed start model_id=%s protocol=%s name=%s count=%d batch_size=%d",
            self.config.id,
            self.config.protocol,
            self.config.name,
            len(texts),
            self.batch_size,
        )
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                vectors.extend(self._adapter.embed(batch))
        except Exception as exc:
            log.error(
                "embed failed model_id=%s protocol=%s error=%s",
                self.config.id,
                self.config.protocol,
                exc,
            )
            raise
        log.info("embed done model_id=%s count=%d", self.config.id, len(vectors))
        return vectors
