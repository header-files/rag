"""Protocol adapter abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from common.config.models import ModelConfig
from llm.transport.http import HttpTransport
from llm.types import ChatRequest, ChatResponse, StreamChunk


class ProtocolAdapter(ABC):
    def __init__(self, config: ModelConfig, transport: HttpTransport | None = None) -> None:
        self.config = config
        self.transport = transport or HttpTransport(timeout=config.timeout_seconds)

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        raise NotImplementedError

    @abstractmethod
    async def achat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    async def achat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        yield  # pragma: no cover
