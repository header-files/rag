"""Vector store abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.types import DocumentRecord, SearchHit


class VectorStore(ABC):
    @abstractmethod
    def add_document(
        self,
        document: DocumentRecord,
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str) -> DocumentRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[DocumentRecord]:
        raise NotImplementedError

    @abstractmethod
    def rebuild(self) -> None:
        raise NotImplementedError
