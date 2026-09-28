"""RAG domain errors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RagError(Exception):
    message: str
    type: str = "rag_error"

    def __str__(self) -> str:
        return self.message
