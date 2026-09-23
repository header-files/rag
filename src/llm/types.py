"""Unified client-facing types (protocol-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]
StreamChunkType = Literal["reasoning", "content", "finish"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ChatRequest:
    messages: list[Message]
    temperature: float | None = None
    max_output_tokens: int | None = None
    stop: list[str] | None = None


@dataclass
class ChatResponse:
    content: str
    reasoning: str | None = None
    finish_reason: FinishReason | None = None
    model: str | None = None
    usage: Usage | None = None
    metadata: dict | None = None


@dataclass
class StreamChunk:
    type: StreamChunkType
    delta: str = ""
    finish_reason: FinishReason | None = None
    model: str | None = None
    usage: Usage | None = None
