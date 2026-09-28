"""FAISS-backed vector store (IndexFlatIP, soft delete, dual-path search)."""

from __future__ import annotations

import threading
import uuid

import faiss
import numpy as np

from common.logger import get_logger
from rag.errors import RagError
from rag.types import ChunkRecord, DocumentRecord, SearchHit
from rag.vectorstore.base import VectorStore

log = get_logger(__name__)


class FaissVectorStore(VectorStore):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dimension: int | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._documents_by_id: dict[str, DocumentRecord] = {}
        self._chunks_by_faiss_id: dict[int, ChunkRecord] = {}
        self._faiss_ids_by_chunk_id: dict[str, int] = {}
        self._faiss_ids_by_document_id: dict[str, set[int]] = {}
        self._vectors_by_faiss_id: dict[int, np.ndarray] = {}
        self._next_faiss_id = 0

    def add_document(
        self,
        document: DocumentRecord,
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        if not texts:
            raise RagError(message="Cannot index a document with no chunks")
        if len(texts) != len(vectors):
            raise RagError(
                message=(
                    f"Chunk/vector count mismatch: texts={len(texts)} "
                    f"vectors={len(vectors)}"
                )
            )
        matrix = _as_normalized_matrix(vectors)
        dimension = matrix.shape[1]
        with self._lock:
            if document.document_id in self._documents_by_id:
                raise RagError(
                    message=f"Document already exists: {document.document_id}"
                )
            self._ensure_index(dimension)
            assert self._index is not None
            start = self._next_faiss_id
            faiss_ids = list(range(start, start + len(texts)))
            self._index.add(matrix)
            self._next_faiss_id = start + len(texts)
            id_set: set[int] = set()
            for offset, (text, faiss_id) in enumerate(zip(texts, faiss_ids, strict=True)):
                chunk_id = uuid.uuid4().hex
                chunk = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=offset,
                    text=text,
                    faiss_id=faiss_id,
                    deleted=False,
                )
                self._chunks_by_faiss_id[faiss_id] = chunk
                self._faiss_ids_by_chunk_id[chunk_id] = faiss_id
                self._vectors_by_faiss_id[faiss_id] = matrix[offset]
                id_set.add(faiss_id)
            self._faiss_ids_by_document_id[document.document_id] = id_set
            self._documents_by_id[document.document_id] = document
            log.info(
                "faiss add document_id=%s chunks=%d dim=%d",
                document.document_id,
                len(texts),
                dimension,
            )

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if top_k < 1:
            raise RagError(message="top_k must be >= 1")
        query = _as_normalized_vector(vector)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            if self._dimension is not None and query.shape[0] != self._dimension:
                raise RagError(
                    message=(
                        "Vector dimension mismatch: "
                        f"query={query.shape[0]} store={self._dimension}"
                    )
                )
            if document_ids is None:
                return self._search_global(query, top_k)
            return self._search_allowed(query, top_k, document_ids)

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents_by_id:
                return False
            for faiss_id in self._faiss_ids_by_document_id.get(document_id, set()):
                chunk = self._chunks_by_faiss_id.get(faiss_id)
                if chunk is not None:
                    chunk.deleted = True
            self._documents_by_id.pop(document_id, None)
            self._faiss_ids_by_document_id.pop(document_id, None)
            log.info("faiss soft-delete document_id=%s", document_id)
            return True

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._documents_by_id.get(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(
                self._documents_by_id.values(),
                key=lambda item: (item.created_at, item.document_id),
            )

    def rebuild(self) -> None:
        with self._lock:
            active: list[tuple[ChunkRecord, np.ndarray]] = []
            documents = dict(self._documents_by_id)
            for document_id, faiss_ids in self._faiss_ids_by_document_id.items():
                if document_id not in documents:
                    continue
                for faiss_id in sorted(faiss_ids):
                    chunk = self._chunks_by_faiss_id.get(faiss_id)
                    vector = self._vectors_by_faiss_id.get(faiss_id)
                    if chunk is None or vector is None or chunk.deleted:
                        continue
                    active.append((chunk, vector))

            self._chunks_by_faiss_id.clear()
            self._faiss_ids_by_chunk_id.clear()
            self._faiss_ids_by_document_id.clear()
            self._vectors_by_faiss_id.clear()
            self._next_faiss_id = 0
            self._index = None
            self._dimension = None

            if not active:
                log.info("faiss rebuild empty")
                return

            dimension = int(active[0][1].shape[0])
            self._ensure_index(dimension)
            assert self._index is not None
            matrix = np.vstack([vector for _, vector in active]).astype(np.float32)
            self._index.add(matrix)
            for offset, (chunk, vector) in enumerate(active):
                faiss_id = offset
                updated = ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    faiss_id=faiss_id,
                    deleted=False,
                )
                self._chunks_by_faiss_id[faiss_id] = updated
                self._faiss_ids_by_chunk_id[updated.chunk_id] = faiss_id
                self._vectors_by_faiss_id[faiss_id] = vector
                self._faiss_ids_by_document_id.setdefault(
                    updated.document_id, set()
                ).add(faiss_id)
            self._next_faiss_id = len(active)
            log.info("faiss rebuild vectors=%d dim=%d", len(active), dimension)

    def _ensure_index(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension
            self._index = faiss.IndexFlatIP(dimension)
            return
        if dimension != self._dimension:
            raise RagError(
                message=(
                    "Vector dimension mismatch: "
                    f"store={self._dimension} document={dimension}"
                )
            )
        if self._index is None:
            self._index = faiss.IndexFlatIP(dimension)

    def _search_global(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        assert self._index is not None
        active_ids = [
            faiss_id
            for faiss_id, chunk in self._chunks_by_faiss_id.items()
            if not chunk.deleted and chunk.document_id in self._documents_by_id
        ]
        if not active_ids:
            return []
        # Prefer FAISS search; refill from active set if soft-deletes create holes.
        request_k = min(self._index.ntotal, max(top_k * 4, top_k))
        scores, indices = self._index.search(query.reshape(1, -1), request_k)
        hits: list[SearchHit] = []
        seen: set[int] = set()
        for score, faiss_id in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if faiss_id < 0 or faiss_id in seen:
                continue
            hit = self._hit_from_faiss_id(faiss_id, float(score))
            if hit is None:
                continue
            seen.add(faiss_id)
            hits.append(hit)
            if len(hits) >= top_k:
                return hits
        if len(hits) >= top_k:
            return hits[:top_k]
        # Exact refill over remaining active vectors.
        remaining = [faiss_id for faiss_id in active_ids if faiss_id not in seen]
        if remaining:
            refill = self._score_allowed(query, remaining, top_k - len(hits))
            hits.extend(refill)
        return hits[:top_k]

    def _search_allowed(
        self,
        query: np.ndarray,
        top_k: int,
        document_ids: list[str],
    ) -> list[SearchHit]:
        allowed: list[int] = []
        for document_id in document_ids:
            for faiss_id in self._faiss_ids_by_document_id.get(document_id, set()):
                chunk = self._chunks_by_faiss_id.get(faiss_id)
                if chunk is None or chunk.deleted:
                    continue
                if chunk.document_id not in self._documents_by_id:
                    continue
                allowed.append(faiss_id)
        if not allowed:
            return []
        return self._score_allowed(query, allowed, top_k)

    def _score_allowed(
        self,
        query: np.ndarray,
        faiss_ids: list[int],
        top_k: int,
    ) -> list[SearchHit]:
        matrix = np.vstack(
            [self._vectors_by_faiss_id[faiss_id] for faiss_id in faiss_ids]
        ).astype(np.float32)
        scores = matrix @ query
        order = np.argsort(-scores)[:top_k]
        hits: list[SearchHit] = []
        for offset in order.tolist():
            faiss_id = faiss_ids[offset]
            hit = self._hit_from_faiss_id(faiss_id, float(scores[offset]))
            if hit is not None:
                hits.append(hit)
        return hits

    def _hit_from_faiss_id(self, faiss_id: int, score: float) -> SearchHit | None:
        chunk = self._chunks_by_faiss_id.get(faiss_id)
        if chunk is None or chunk.deleted:
            return None
        document = self._documents_by_id.get(chunk.document_id)
        if document is None:
            return None
        return SearchHit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=document.filename,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=score,
        )


def _as_normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise RagError(message="Vectors must be a non-empty 2D matrix")
    return _l2_normalize_rows(matrix)


def _as_normalized_vector(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RagError(message="Query vector must not be empty")
    return _l2_normalize_rows(array.reshape(1, -1))[0]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms
