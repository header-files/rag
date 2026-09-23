"""OpenAI Chat Completions protocol adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from llm.errors import LLMError
from llm.protocols.base import ProtocolAdapter
from llm.protocols.common import (
    iter_sse_data_lines,
    join_url,
    map_finish_reason,
    parse_json_object,
    parse_usage_openai_chat,
    raise_from_http_body,
)
from llm.types import ChatRequest, ChatResponse, StreamChunk


class OpenAICompletionsAdapter(ProtocolAdapter):
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url
        if base.endswith("/chat/completions"):
            return base
        return join_url(base, "/chat/completions")

    def _build_body(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.name,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "stream": stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        if request.stop is not None:
            body["stop"] = request.stop
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or []
        content = ""
        reasoning: str | None = None
        finish_reason = None
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning is not None:
                reasoning = str(reasoning)
            finish_reason = map_finish_reason(choice.get("finish_reason"))
        return ChatResponse(
            content=content if isinstance(content, str) else str(content),
            reasoning=reasoning,
            finish_reason=finish_reason,
            model=data.get("model") or self.config.name,
            usage=parse_usage_openai_chat(data.get("usage")),
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        response = self.transport.request(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=False),
        )
        if response.status_code >= 400:
            raise_from_http_body(
                response.status_code, response.text, protocol="openai-completions"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise LLMError(type="api_error", message="Invalid JSON response")
        return self._parse_response(data)

    async def achat(self, request: ChatRequest) -> ChatResponse:
        response = await self.transport.arequest(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=False),
        )
        if response.status_code >= 400:
            raise_from_http_body(
                response.status_code, response.text, protocol="openai-completions"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise LLMError(type="api_error", message="Invalid JSON response")
        return self._parse_response(data)

    def _iter_chunks(self, lines: Iterator[str]) -> Iterator[StreamChunk]:
        saw_finish = False
        last_model: str | None = None
        last_usage = None
        last_finish = None
        try:
            for data_line in iter_sse_data_lines(lines):
                payload = parse_json_object(data_line)
                if not payload:
                    continue
                if payload.get("model"):
                    last_model = payload["model"]
                usage = parse_usage_openai_chat(payload.get("usage"))
                if usage:
                    last_usage = usage
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield StreamChunk(type="reasoning", delta=str(reasoning))
                    text = delta.get("content")
                    if text:
                        yield StreamChunk(type="content", delta=str(text))
                    fr = map_finish_reason(choice.get("finish_reason"))
                    if fr is not None:
                        last_finish = fr
            saw_finish = True
            yield StreamChunk(
                type="finish",
                delta="",
                finish_reason=last_finish,
                model=last_model or self.config.name,
                usage=last_usage,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                type="stream_error",
                message=str(exc) or "Stream parsing failed",
                retryable=None,
            ) from exc
        finally:
            if not saw_finish:
                # mid-stream failure: do not emit finish (caller handles Error)
                pass

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        lines = self.transport.stream(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=True),
        )
        yield from self._iter_chunks(lines)

    async def achat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        lines = self.transport.astream(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=True),
        )

        async def _agen() -> AsyncIterator[str]:
            async for line in lines:
                yield line

        # Buffer async lines into sync iterator via list for reuse of parser
        collected: list[str] = []
        async for line in _agen():
            collected.append(line)
        for chunk in self._iter_chunks(iter(collected)):
            yield chunk
