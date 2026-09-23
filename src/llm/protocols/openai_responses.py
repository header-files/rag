"""OpenAI Responses API protocol adapter."""

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


def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
            continue
        role = "user" if msg.role == "user" else "assistant"
        input_items.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": msg.content}]
                if role == "user"
                else [{"type": "output_text", "text": msg.content}],
            }
        )
    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, input_items


def _extract_text_from_output(output: list[Any]) -> tuple[str, str | None]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            for part in item.get("summary") or item.get("content") or []:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        reasoning_parts.append(str(text))
                elif isinstance(part, str):
                    reasoning_parts.append(part)
            continue
        if item_type in ("message", None):
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                text = part.get("text")
                if text is None:
                    continue
                if ptype in ("output_text", "text"):
                    content_parts.append(str(text))
                elif ptype in ("reasoning", "reasoning_text"):
                    reasoning_parts.append(str(text))
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts) or None
    return content, reasoning


class OpenAIResponsesAdapter(ProtocolAdapter):
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url
        if base.endswith("/responses"):
            return base
        return join_url(base, "/responses")

    def _build_body(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        instructions, input_items = _split_system(request.messages)
        body: dict[str, Any] = {
            "model": self.config.name,
            "input": input_items,
            "stream": stream,
        }
        if instructions:
            body["instructions"] = instructions
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
        if request.stop is not None:
            # Responses API may use different naming; pass through as stop
            body["stop"] = request.stop
        return body

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        output = data.get("output") or []
        content, reasoning = _extract_text_from_output(output)
        status = data.get("status")
        finish = None
        if status == "completed":
            finish = "stop"
        elif status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            finish = map_finish_reason(reason) or "length"
        return ChatResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=finish,
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
                response.status_code, response.text, protocol="openai-responses"
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
                response.status_code, response.text, protocol="openai-responses"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise LLMError(type="api_error", message="Invalid JSON response")
        return self._parse_response(data)

    def _iter_chunks(self, lines: Iterator[str]) -> Iterator[StreamChunk]:
        last_model: str | None = None
        last_usage = None
        last_finish: str | None = None
        try:
            for data_line in iter_sse_data_lines(lines):
                payload = parse_json_object(data_line)
                if not payload:
                    continue
                event_type = payload.get("type") or ""
                if payload.get("model"):
                    last_model = payload["model"]

                # Nested response object on completed events
                response_obj = payload.get("response")
                if isinstance(response_obj, dict):
                    if response_obj.get("model"):
                        last_model = response_obj["model"]
                    usage = parse_usage_tokens(response_obj.get("usage"))
                    if usage:
                        last_usage = usage

                if event_type in (
                    "response.output_text.delta",
                    "response.content_part.delta",
                ):
                    delta = payload.get("delta")
                    if isinstance(delta, str) and delta:
                        yield StreamChunk(type="content", delta=delta)
                    elif isinstance(delta, dict):
                        text = delta.get("text") or delta.get("value")
                        if text:
                            yield StreamChunk(type="content", delta=str(text))
                elif event_type in (
                    "response.reasoning_text.delta",
                    "response.reasoning.delta",
                ):
                    delta = payload.get("delta")
                    if isinstance(delta, str) and delta:
                        yield StreamChunk(type="reasoning", delta=delta)
                    elif isinstance(delta, dict) and delta.get("text"):
                        yield StreamChunk(type="reasoning", delta=str(delta["text"]))
                elif event_type == "response.completed":
                    last_finish = "stop"
                    if isinstance(response_obj, dict):
                        last_usage = parse_usage_tokens(response_obj.get("usage")) or last_usage
                elif event_type == "response.incomplete":
                    last_finish = "length"
                elif event_type == "error":
                    raise LLMError(
                        type="stream_error",
                        message=str(payload.get("message") or payload),
                        retryable=None,
                    )

            yield StreamChunk(
                type="finish",
                delta="",
                finish_reason=last_finish,  # type: ignore[arg-type]
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
