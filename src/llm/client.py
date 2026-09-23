"""Unified LLM client entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from common.config.models import ModelConfig, get_model, load_models
from llm.protocols.registry import create_adapter
from llm.transport.http import HttpTransport
from llm.types import ChatRequest, ChatResponse, StreamChunk


class LLMClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        self._adapter = create_adapter(config, transport=transport)

    @classmethod
    def from_env(
        cls,
        model_id: str | None = None,
        *,
        env_path: str | Path | None = None,
    ) -> LLMClient:
        config = get_model(model_id, env_path=env_path)
        return cls(config)

    @classmethod
    def list_models(
        cls,
        *,
        env_path: str | Path | None = None,
    ) -> tuple[str, dict[str, ModelConfig]]:
        return load_models(env_path)

    def chat(self, request: ChatRequest) -> ChatResponse:
        return self._adapter.chat(request)

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        return self._adapter.chat_stream(request)

    async def achat(self, request: ChatRequest) -> ChatResponse:
        return await self._adapter.achat(request)

    async def achat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        async for chunk in self._adapter.achat_stream(request):
            yield chunk
