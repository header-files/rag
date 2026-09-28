"""Document ingestion."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from common.config.rag import RagSettings
from common.logger import get_logger
from rag.chunking import split_text
from rag.embedding.client import EmbeddingClient
from rag.errors import RagError
from rag.parsers import parse_document
from rag.types import DocumentRecord
from rag.vectorstore.base import VectorStore

log = get_logger(__name__)


class IngestService:
    def __init__(
        self,
        settings: RagSettings,
        store: VectorStore,
        embedding: EmbeddingClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedding = embedding

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        media_type: str | None = None,
    ) -> DocumentRecord:
        name = Path(filename).name
        if not name or name in {".", ".."}:
            raise RagError(message="Invalid filename")
        log.info(
            "ingest start filename=%s bytes=%d media_type=%s",
            name,
            len(data),
            media_type or "-",
        )
        try:
            if not data:
                raise RagError(message="Empty file", type="empty_document")
            if len(data) > self.settings.max_upload_bytes:
                raise RagError(
                    message=(
                        f"File exceeds RAG_MAX_UPLOAD_BYTES "
                        f"({self.settings.max_upload_bytes})"
                    ),
                    type="payload_too_large",
                )
            parsed = parse_document(data, name)
            chunks = split_text(
                parsed.text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=self.settings.separators,
            )
            if not chunks:
                raise RagError(
                    message=f"Document '{name}' has no text content",
                    type="empty_document",
                )
            vectors = self.embedding.embed(chunks)
            document = DocumentRecord(
                document_id=uuid.uuid4().hex,
                filename=name,
                media_type=media_type,
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=dict(parsed.metadata),
            )
            self.store.add_document(document, chunks, vectors)
        except Exception as exc:
            log.error("ingest failed filename=%s error=%s", name, exc)
            raise
        log.info(
            "ingest done document_id=%s filename=%s chunks=%d",
            document.document_id,
            document.filename,
            document.chunk_count,
        )
        return document
