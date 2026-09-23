"""Tests for OpenAI Responses adapter (mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import respx

from llm.protocols.openai_responses import OpenAIResponsesAdapter
from common.config.models import ModelConfig
from llm.types import ChatRequest, Message


@respx.mock
def test_responses_non_stream() -> None:
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"text": "why"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}],
                    },
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )
    )
    adapter = OpenAIResponsesAdapter(
        ModelConfig(
            id="r",
            protocol="openai-responses",
            name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="sk",
        )
    )
    resp = adapter.chat(
        ChatRequest(
            messages=[
                Message(role="system", content="s"),
                Message(role="user", content="u"),
            ],
            max_output_tokens=64,
        )
    )
    body = json.loads(route.calls.last.request.content.decode())
    assert body["instructions"] == "s"
    assert body["max_output_tokens"] == 64
    assert resp.content == "hi"
    assert resp.reasoning == "why"
    assert resp.finish_reason == "stop"
