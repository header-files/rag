"""Vector store registry."""

from __future__ import annotations

from common.config.errors import ConfigError
from rag.vectorstore.base import VectorStore
from rag.vectorstore.faiss_store import FaissVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {
    "faiss": FaissVectorStore,
}


def create_vector_store(kind: str) -> VectorStore:
    normalized = kind.strip().lower()
    cls = _REGISTRY.get(normalized)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ConfigError(
            message=f"Unsupported vector store '{kind}'. Available: {known}"
        )
    return cls()
