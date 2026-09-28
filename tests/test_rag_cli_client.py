"""CLI HTTP client tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from rag.cli.client import ApiClientError, RagApiClient


@respx.mock
def test_upload_ask_session(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")
    respx.post("http://rag.test/documents").mock(
        return_value=httpx.Response(
            201, json={"id": "doc1", "filename": "note.txt", "chunk_count": 1}
        )
    )
    respx.post("http://rag.test/sessions").mock(
        return_value=httpx.Response(200, json={"session_id": "s1"})
    )
    respx.post("http://rag.test/chat").mock(
        return_value=httpx.Response(
            200, json={"answer": "hi", "citations": [], "session_id": "s1"}
        )
    )
    client = RagApiClient("http://rag.test")
    assert client.upload(path)["id"] == "doc1"
    assert client.create_session() == "s1"
    assert client.ask("q", session_id="s1")["answer"] == "hi"


@respx.mock
def test_stream_and_error() -> None:
    respx.get("http://rag.test/documents").mock(
        return_value=httpx.Response(
            404, json={"error": {"type": "not_found", "message": "missing"}}
        )
    )
    client = RagApiClient("http://rag.test")
    with pytest.raises(ApiClientError, match="missing"):
        client.list_documents()
    respx.post("http://rag.test/chat").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'event: session\ndata: {"session_id": "s1"}\n\n'
                'event: content\ndata: {"delta": "ok"}\n\n'
            ),
        )
    )
    events = list(client.ask_stream("q"))
    assert events[0][0] == "session"
    assert events[1] == ("content", {"delta": "ok"})
