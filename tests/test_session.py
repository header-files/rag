"""Tests for ChatSession context handling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from llm.session import ChatSession
from common.config.models import ModelConfig
from llm.types import ChatRequest, ChatResponse, Message, StreamChunk


class _FakeClient:
    def __init__(self) -> None:
        self.config = ModelConfig(
            id="fake",
            protocol="openai-completions",
            name="fake",
            base_url="https://example.com",
            api_key="k",
        )
        self.calls: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(content="pong", finish_reason="stop")

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.calls.append(request)
        yield StreamChunk(type="content", delta="po")
        yield StreamChunk(type="content", delta="ng")
        yield StreamChunk(type="finish", finish_reason="stop")


def test_session_appends_on_success() -> None:
    client: Any = _FakeClient()
    session = ChatSession(client, system_prompt="sys")
    resp = session.send("ping")
    assert resp.content == "pong"
    assert [m.role for m in session.messages] == ["system", "user", "assistant"]
    assert session.messages[-1].content == "pong"
    assert len(client.calls[0].messages) == 2  # system + user


def test_session_stream_success() -> None:
    client: Any = _FakeClient()
    session = ChatSession(client)
    chunks = list(session.send_stream("ping"))
    assert chunks[-1].type == "finish"
    assert session.messages[-1].content == "pong"


def test_session_stream_failure_keeps_no_assistant() -> None:
    class Boom(_FakeClient):
        def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
            yield StreamChunk(type="content", delta="partial")
            raise RuntimeError("boom")

    session = ChatSession(Boom())
    try:
        list(session.send_stream("ping"))
    except RuntimeError:
        pass
    assert len(session.messages) == 0


def test_switch_client_clears_history() -> None:
    a: Any = _FakeClient()
    b: Any = _FakeClient()
    b.config = ModelConfig(
        id="other",
        protocol="anthropic-messages",
        name="x",
        base_url="https://example.com",
        api_key="k",
    )
    session = ChatSession(a, system_prompt="sys")
    session.send("hi")
    session.switch_client(b, clear_history=True)
    assert [m.role for m in session.messages] == ["system"]
