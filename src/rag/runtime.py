"""Wire config and services into one runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config.rag import RagSettings, load_rag_settings
from common.logger import get_logger
from llm.client import LLMClient
from rag.context.builder import ContextBuilder
from rag.embedding.client import EmbeddingClient
from rag.ingest import IngestService
from rag.query.service import QueryService
from rag.rerank.client import RerankClient
from rag.retrieve.retriever import Retriever
from rag.rewrite.rewriter import QueryRewriter
from rag.session.store import SessionStore
from rag.vectorstore.base import VectorStore
from rag.vectorstore.registry import create_vector_store

log = get_logger(__name__)


@dataclass
class RagRuntime:
    settings: RagSettings
    store: VectorStore
    embedding: EmbeddingClient
    llm: LLMClient
    ingest: IngestService
    query: QueryService
    sessions: SessionStore
    rerank: RerankClient | None = None
    rewrite_llm: LLMClient | None = None

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> RagRuntime:
        settings = load_rag_settings(env_path)
        store = create_vector_store(settings.vector_store)
        embedding = EmbeddingClient.from_env(
            settings.embedding_id,
            env_path=env_path,
            batch_size=settings.embed_batch_size,
        )
        rerank = None
        if settings.rerank_enabled:
            rerank = RerankClient.from_env(settings.rerank_id, env_path=env_path)
        llm = LLMClient.from_env(settings.chat_model_id, env_path=env_path)
        rewrite_llm = None
        rewriter = None
        if settings.query_rewrite_enabled:
            if settings.query_rewrite_model_id:
                rewrite_llm = LLMClient.from_env(
                    settings.query_rewrite_model_id,
                    env_path=env_path,
                )
            else:
                rewrite_llm = llm
            rewriter = QueryRewriter(
                rewrite_llm,
                prompt=settings.query_rewrite_prompt,
            )
        sessions = SessionStore(max_history_turns=settings.max_history_turns)
        ingest = IngestService(settings, store, embedding)
        retriever = Retriever(settings, store, embedding, rerank=rerank)
        context_builder = ContextBuilder(
            system_prompt=settings.system_prompt,
            max_context_chars=settings.max_context_chars,
        )
        query = QueryService(
            settings,
            store,
            retriever,
            context_builder,
            llm,
            sessions,
            rewriter=rewriter,
        )
        log.info(
            "RAG runtime ready vector_store=%s embedding=%s rerank=%s "
            "chat_model=%s query_rewrite=%s rewrite_model=%s",
            settings.vector_store,
            embedding.config.id,
            rerank.config.id if rerank else "-",
            llm.config.id,
            settings.query_rewrite_enabled,
            rewrite_llm.config.id if rewrite_llm is not None else "-",
        )
        return cls(
            settings=settings,
            store=store,
            embedding=embedding,
            llm=llm,
            ingest=ingest,
            query=query,
            sessions=sessions,
            rerank=rerank,
            rewrite_llm=rewrite_llm,
        )
