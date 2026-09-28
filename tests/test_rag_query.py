"""Query / context / session tests."""

from __future__ import annotations

from collections.abc import Iterator

from common.config.rag import RagSettings
from llm.types import ChatRequest, ChatResponse, Message, StreamChunk
from rag.context.builder import ContextBuilder
from rag.ingest import IngestService
from rag.query.service import QueryService
from rag.rerank.base import RerankResult
from rag.retrieve.retriever import Retriever
from rag.rewrite.rewriter import QueryRewriter
from rag.session.store import SessionStore
from rag.types import SearchHit
from rag.vectorstore.faiss_store import FaissVectorStore


class _Embed:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[1.0, 0.0] if "widgets" in text else [0.0, 1.0] for text in texts]


class _Rerank:
    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None):
        del query
        ranked = sorted(
            range(len(documents)),
            key=lambda index: ("widgets" not in documents[index], index),
        )
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            RerankResult(index=index, score=1.0 - offset)
            for offset, index in enumerate(ranked)
        ]


class _LLM:
    def __init__(self, *, rewrite_to: str | None = None) -> None:
        self.last: ChatRequest | None = None
        self.rewrite_to = rewrite_to
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.last = request
        self.calls += 1
        last = request.messages[-1].content if request.messages else ""
        if "改写下面的最新问题为独立检索查询" in last:
            return ChatResponse(
                content=self.rewrite_to or "where are the widgets",
                finish_reason="stop",
                model="fake-rewrite",
            )
        return ChatResponse(content="答案[1]", finish_reason="stop", model="fake")

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.last = request
        yield StreamChunk(type="content", delta="答案")
        yield StreamChunk(type="finish", finish_reason="stop")


def _services(*, rewriter: QueryRewriter | None = None, llm: _LLM | None = None):
    settings = RagSettings(
        chunk_size=64,
        chunk_overlap=0,
        retrieve_top_k=5,
        rerank_enabled=True,
        rerank_top_n=1,
        query_rewrite_enabled=rewriter is not None,
        max_context_chars=2000,
        max_history_turns=2,
        system_prompt="sys",
    )
    store = FaissVectorStore()
    embed = _Embed()
    llm = llm or _LLM()
    sessions = SessionStore(max_history_turns=settings.max_history_turns)
    ingest = IngestService(settings, store, embed)  # type: ignore[arg-type]
    retriever = Retriever(settings, store, embed, rerank=_Rerank())  # type: ignore[arg-type]
    builder = ContextBuilder(
        system_prompt=settings.system_prompt,
        max_context_chars=settings.max_context_chars,
    )
    query = QueryService(
        settings,
        store,
        retriever,
        builder,
        llm,  # type: ignore[arg-type]
        sessions,
        rewriter=rewriter,
    )
    return ingest, query, llm, sessions, embed


def test_ingest_multiturn_and_history_keeps_query_only() -> None:
    ingest, query, llm, sessions, _ = _services()
    ingest.ingest_bytes(
        b"widgets live in the dock\n\nunrelated orchard notes",
        "guide.txt",
    )
    first = query.prepare("where are the widgets")
    answer = query.complete(first)
    assert answer.content == "答案[1]"
    assert llm.last is not None
    assert "## 资料" in llm.last.messages[-1].content
    assert "## 问题" in llm.last.messages[-1].content
    assert "widgets" in llm.last.messages[-1].content

    session = sessions.get(first.session_id)
    assert session.messages[0].content == "where are the widgets"
    assert "## 资料" not in session.messages[0].content

    second = query.prepare("say more", session_id=first.session_id)
    query.complete(second)
    assert llm.last is not None
    assert any(m.role == "assistant" for m in llm.last.messages[:-1])


def test_context_builder_partitions() -> None:
    builder = ContextBuilder(system_prompt="sys", max_context_chars=1000)
    messages, used = builder.build(
        query="q",
        history=[Message(role="user", content="old"), Message(role="assistant", content="a")],
        hits=[
            SearchHit(
                chunk_id="c1",
                document_id="d1",
                filename="f.txt",
                chunk_index=0,
                text="body",
                score=1.0,
            )
        ],
    )
    assert used[0].text == "body"
    assert messages[0].role == "system"
    assert messages[1].content == "old"
    assert "## 问题\nq" in messages[-1].content


def test_empty_store_skips_llm() -> None:
    _, query, llm, _, _ = _services()
    prepared = query.prepare("anything")
    answer = query.complete(prepared)
    assert "没有可用资料" in answer.content
    assert llm.last is None


def test_query_rewrite_uses_rewritten_for_retrieve_keeps_original_in_context() -> None:
    rewrite_llm = _LLM(rewrite_to="where are the widgets")
    rewriter = QueryRewriter(rewrite_llm)  # type: ignore[arg-type]
    ingest, query, _, _, embed = _services(rewriter=rewriter, llm=_LLM())
    ingest.ingest_bytes(b"widgets live in the dock", "guide.txt")
    first = query.prepare("where are the widgets")
    query.complete(first)
    assert first.retrieval_query == "where are the widgets"

    second = query.prepare("它在哪里", session_id=first.session_id)
    assert second.retrieval_query == "where are the widgets"
    assert embed.queries[-1] == "where are the widgets"
    assert "## 问题\n它在哪里" in second.messages[-1].content
    assert rewrite_llm.calls >= 1
