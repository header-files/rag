"""HTTP API tests."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from api import create_app
from common.config.rag import RagSettings
from llm.types import ChatRequest, ChatResponse, StreamChunk
from rag.context.builder import ContextBuilder
from rag.ingest import IngestService
from rag.query.service import QueryService
from rag.retrieve.retriever import Retriever
from rag.runtime import RagRuntime
from rag.session.store import SessionStore
from rag.vectorstore.faiss_store import FaissVectorStore


class _Embed:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "widgets" in text else [0.0, 1.0] for text in texts]


class _LLM:
    def chat(self, request: ChatRequest) -> ChatResponse:
        del request
        return ChatResponse(content="在码头", finish_reason="stop", model="fake")

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        del request
        yield StreamChunk(type="content", delta="在码头")
        yield StreamChunk(type="finish", finish_reason="stop", model="fake")


def _client(max_upload_bytes: int = 1024) -> TestClient:
    settings = RagSettings(
        chunk_size=80,
        chunk_overlap=0,
        rerank_enabled=False,
        max_upload_bytes=max_upload_bytes,
        system_prompt="sys",
    )
    store = FaissVectorStore()
    embed = _Embed()
    llm = _LLM()
    sessions = SessionStore()
    retriever = Retriever(settings, store, embed)  # type: ignore[arg-type]
    runtime = RagRuntime(
        settings=settings,
        store=store,
        embedding=embed,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        ingest=IngestService(settings, store, embed),  # type: ignore[arg-type]
        query=QueryService(
            settings,
            store,
            retriever,
            ContextBuilder(system_prompt="sys", max_context_chars=2000),
            llm,  # type: ignore[arg-type]
            sessions,
        ),
        sessions=sessions,
    )
    return TestClient(create_app(runtime))


def test_upload_session_chat_stream() -> None:
    client = _client()
    assert client.get("/health").json()["documents"] == 0
    uploaded = client.post(
        "/documents",
        files={"file": ("guide.txt", b"widgets live in the dock", "text/plain")},
    )
    assert uploaded.status_code == 201
    document_id = uploaded.json()["id"]

    session = client.post("/sessions").json()["session_id"]
    answer = client.post(
        "/chat",
        json={"message": "where are widgets", "session_id": session},
    )
    assert answer.status_code == 200
    body = answer.json()
    assert body["answer"] == "在码头"
    assert body["session_id"] == session
    assert body["citations"][0]["document_id"] == document_id

    with client.stream(
        "POST",
        "/chat",
        json={"message": "again", "session_id": session, "stream": True},
    ) as response:
        raw = response.read().decode()
    assert "event: session" in raw
    assert "event: citations" in raw
    assert "码头" in raw

    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert client.delete(f"/sessions/{session}").status_code == 204


def test_upload_rejects() -> None:
    client = _client(max_upload_bytes=4)
    assert (
        client.post(
            "/documents",
            files={"file": ("a.exe", b"abcd", "application/octet-stream")},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/documents",
            files={"file": ("a.txt", b"12345", "text/plain")},
        ).status_code
        == 413
    )
