"""Retrieve: embed → vector search → optional rerank."""

from __future__ import annotations

from common.config.rag import RagSettings
from common.logger import get_logger
from rag.embedding.client import EmbeddingClient
from rag.errors import RagError
from rag.rerank.client import RerankClient
from rag.types import SearchHit
from rag.vectorstore.base import VectorStore

log = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        settings: RagSettings,
        store: VectorStore,
        embedding: EmbeddingClient,
        rerank: RerankClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedding = embedding
        self.rerank = rerank

    def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        text = query.strip()
        if not text:
            raise RagError(message="Query must not be empty")
        vectors = self.embedding.embed([text])
        if len(vectors) != 1:
            raise RagError(message="Embedding provider returned unexpected vector count")
        hits = self.store.search(
            vectors[0],
            top_k=self.settings.retrieve_top_k,
            document_ids=document_ids,
        )
        log.info("retrieve vector hits=%d", len(hits))
        if not hits or self.rerank is None:
            return hits
        results = self.rerank.rerank(
            text,
            [hit.text for hit in hits],
            top_n=self.settings.rerank_top_n,
        )
        ordered: list[SearchHit] = []
        for item in results:
            if item.index < 0 or item.index >= len(hits):
                raise RagError(message=f"Rerank returned invalid index {item.index}")
            source = hits[item.index]
            ordered.append(
                SearchHit(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    filename=source.filename,
                    chunk_index=source.chunk_index,
                    text=source.text,
                    score=item.score,
                )
            )
        log.info("retrieve reranked hits=%d", len(ordered))
        return ordered
