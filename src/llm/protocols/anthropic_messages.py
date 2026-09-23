"""Anthropic Messages protocol adapter."""

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
    parse_usage_tokens,
    raise_from_http_body,
)
from llm.types import ChatRequest, ChatResponse, Message, StreamChunk


def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            converted.append({"role": msg.role, "content": msg.content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


def _extract_content_blocks(content: Any) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content, None
    texts: list[str] = []
    reasoning: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                texts.append(str(block.get("text") or ""))
            elif btype in ("thinking", "redacted_thinking"):
                thinking = block.get("thinking") or block.get("data") or ""
                if thinking:
                    reasoning.append(str(thinking))
    return "".join(texts), ("".join(reasoning) or None)


class AnthropicMessagesAdapter(ProtocolAdapter):
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.config.anthropic_version,
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return join_url(base, "/messages")
        return join_url(base, "/v1/messages")

    def _build_body(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = _split_system(request.messages)
        body: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.max_output_tokens
            if request.max_output_tokens is not None
            else 4096,
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop is not None:
            body["stop_sequences"] = request.stop
        return body

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        content, reasoning = _extract_content_blocks(data.get("content"))
        return ChatResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=map_finish_reason(data.get("stop_reason")),
            model=data.get("model") or self.config.name,
            usage=parse_usage_tokens(data.get("usage")),
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
                response.status_code, response.text, protocol="anthropic-messages"
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
                response.status_code, response.text, protocol="anthropic-messages"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise LLMError(type="api_error", message="Invalid JSON response")
        return self._parse_response(data)

    def _iter_chunks(self, lines: Iterator[str]) -> Iterator[StreamChunk]:
        last_model: str | None = None
        last_usage = None
        last_finish = None
        # Anthropic SSE includes event: lines; we only get data: from iter_sse_data_lines
        # Track block type via content_block_start
        current_block_type: str | None = None
        try:
            for data_line in iter_sse_data_lines(lines):
                payload = parse_json_object(data_line)
                if not payload:
                    continue
                event_type = payload.get("type")
                if event_type == "message_start":
                    message = payload.get("message") or {}
                    last_model = message.get("model") or last_model
                    usage = parse_usage_tokens(message.get("usage"))
                    if usage:
                        last_usage = usage
                elif event_type == "content_block_start":
                    block = payload.get("content_block") or {}
                    current_block_type = block.get("type")
                    if current_block_type == "thinking" and block.get("thinking"):
                        yield StreamChunk(
                            type="reasoning", delta=str(block["thinking"])
                        )
                    if current_block_type == "text" and block.get("text"):
                        yield StreamChunk(type="content", delta=str(block["text"]))
                elif event_type == "content_block_delta":
                    delta = payload.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "thinking_delta" or current_block_type == "thinking":
                        text = delta.get("thinking") or delta.get("text")
                        if text:
                            yield StreamChunk(type="reasoning", delta=str(text))
                    elif dtype == "text_delta" or current_block_type == "text":
                        text = delta.get("text")
                        if text:
                            yield StreamChunk(type="content", delta=str(text))
                elif event_type == "content_block_stop":
                    current_block_type = None
                elif event_type == "message_delta":
                    delta = payload.get("delta") or {}
                    last_finish = map_finish_reason(delta.get("stop_reason"))
                    usage = parse_usage_tokens(payload.get("usage"))
                    if usage:
                        # Anthropic message_delta usage often only has output_tokens
                        if last_usage and usage.input_tokens is None:
                            usage.input_tokens = last_usage.input_tokens
                        last_usage = usage
                elif event_type == "message_stop":
                    pass
                elif event_type == "error":
                    err = payload.get("error") or {}
                    raise LLMError(
                        type="stream_error",
                        message=str(err.get("message") or payload),
                        provider_code=str(err["type"]) if err.get("type") else None,
                        retryable=None,
                    )

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

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        lines = self.transport.stream(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=True),
        )
        yield from self._iter_chunks(lines)

    async def achat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        lines_iter = self.transport.astream(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._build_body(request, stream=True),
        )
        buffer: list[str] = []
        async for line in lines_iter:
            buffer.append(line)
        for chunk in self._iter_chunks(iter(buffer)):
            yield chunk
