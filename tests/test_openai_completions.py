"""Integration-style tests for OpenAI Completions adapter (mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm.errors import LLMError
from llm.protocols.openai_completions import OpenAICompletionsAdapter
from common.config.models import ModelConfig
from llm.types import ChatRequest, Message


def _config() -> ModelConfig:
    return ModelConfig(
        id="chatgpt",
        protocol="openai-completions",
        name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )


@respx.mock
def test_chat_non_stream() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "reasoning_content": "think",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        )
    )
    adapter = OpenAICompletionsAdapter(_config())
    resp = adapter.chat(
        ChatRequest(messages=[Message(role="user", content="hi")])
    )
    assert route.called
    body = json.loads(route.calls.last.request.content.decode())
    assert body["model"] == "gpt-4o-mini"
    assert body["stream"] is False
    assert resp.content == "hello"
    assert resp.reasoning == "think"
    assert resp.finish_reason == "stop"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 4


@respx.mock
def test_chat_passes_max_output_tokens_as_max_tokens() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ]
            },
        )
    )
    adapter = OpenAICompletionsAdapter(_config())
    adapter.chat(
        ChatRequest(
            messages=[Message(role="user", content="hi")],
            temperature=0.2,
            max_output_tokens=128,
            stop=["\n"],
        )
    )
    body = json.loads(route.calls.last.request.content.decode())
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128
    assert body["stop"] == ["\n"]


@respx.mock
def test_chat_error_mapping() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unsupported value: 'temperature'",
                    "code": "unsupported_value",
                }
            },
        )
    )
    adapter = OpenAICompletionsAdapter(_config())
    with pytest.raises(LLMError) as exc:
        adapter.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert exc.value.type == "unsupported_parameter"
    assert "temperature" in exc.value.message


@respx.mock
def test_chat_stream() -> None:
    sse = (
        'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"B"},"finish_reason":"stop"}]}\n\n'
        'data: {"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse.encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    adapter = OpenAICompletionsAdapter(_config())
    chunks = list(
        adapter.chat_stream(
            ChatRequest(messages=[Message(role="user", content="hi")])
        )
    )
    types = [c.type for c in chunks]
    assert types[0] == "reasoning"
    assert chunks[0].delta == "r"
    assert types[1] == "content" and chunks[1].delta == "A"
    assert types[2] == "content" and chunks[2].delta == "B"
    assert types[-1] == "finish"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 3
