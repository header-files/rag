"""Tests for Anthropic Messages adapter (mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import respx

from llm.protocols.anthropic_messages import AnthropicMessagesAdapter
from common.config.models import ModelConfig
from llm.types import ChatRequest, Message


def _config() -> ModelConfig:
    return ModelConfig(
        id="claude",
        protocol="anthropic-messages",
        name="claude-3-5-sonnet-latest",
        base_url="https://api.anthropic.com",
        api_key="sk-ant-test",
    )


@respx.mock
def test_chat_splits_system_and_maps_thinking() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "claude-3-5-sonnet-latest",
                "content": [
                    {"type": "thinking", "thinking": "reason"},
                    {"type": "text", "text": "answer"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )
    )
    adapter = AnthropicMessagesAdapter(_config())
    resp = adapter.chat(
        ChatRequest(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="user", content="hi"),
            ],
            max_output_tokens=256,
        )
    )
    body = json.loads(route.calls.last.request.content.decode())
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 256
    assert route.calls.last.request.headers["x-api-key"] == "sk-ant-test"
    assert resp.content == "answer"
    assert resp.reasoning == "reason"
    assert resp.finish_reason == "stop"


@respx.mock
def test_chat_stream_thinking_and_text() -> None:
    sse = (
        'data: {"type":"message_start","message":{"model":"claude-3-5-sonnet-latest","usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        'data: {"type":"content_block_start","content_block":{"type":"thinking","thinking":""}}\n\n'
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"r1"}}\n\n'
        'data: {"type":"content_block_stop"}\n\n'
        'data: {"type":"content_block_start","content_block":{"type":"text","text":""}}\n\n'
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
        'data: {"type":"message_stop"}\n\n'
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    adapter = AnthropicMessagesAdapter(_config())
    chunks = list(
        adapter.chat_stream(
            ChatRequest(messages=[Message(role="user", content="x")])
        )
    )
    assert any(c.type == "reasoning" and c.delta == "r1" for c in chunks)
    assert any(c.type == "content" and c.delta == "hi" for c in chunks)
    assert chunks[-1].type == "finish"
    assert chunks[-1].finish_reason == "stop"
