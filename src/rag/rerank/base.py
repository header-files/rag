"""Rerank protocol adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from common.config.endpoints import EndpointConfig
from llm.transport.http import HttpTransport


@dataclass
class RerankResult:
    index: int
    score: float


class RerankAdapter(ABC):
    def __init__(
        self,
        config: EndpointConfig,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or HttpTransport(timeout=config.timeout_seconds)

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        raise NotImplementedError
