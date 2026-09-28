"""Shared RAG records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    chunk_count: int
    created_at: str
    media_type: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.document_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    faiss_id: int
    deleted: bool = False


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


@dataclass
class Citation:
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "text": self.text,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Answer:
    content: str
    citations: list[Citation]
    session_id: str
    model: str | None = None
    finish_reason: str | None = None
