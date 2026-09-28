"""Embedding protocol adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.config.endpoints import EndpointConfig
from llm.transport.http import HttpTransport


class EmbeddingAdapter(ABC):
    def __init__(
        self,
        config: EndpointConfig,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or HttpTransport(timeout=config.timeout_seconds)

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
