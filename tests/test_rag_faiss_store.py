"""FAISS vector store tests."""

from __future__ import annotations

from rag.types import DocumentRecord
from rag.vectorstore.faiss_store import FaissVectorStore


def _doc(document_id: str, name: str = "a.txt") -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        filename=name,
        chunk_count=1,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_global_search_and_document_filter() -> None:
    store = FaissVectorStore()
    store.add_document(_doc("d1", "alpha.txt"), ["alpha"], [[1.0, 0.0]])
    store.add_document(_doc("d2", "beta.txt"), ["beta"], [[0.0, 1.0]])
    hits = store.search([1.0, 0.1], top_k=2)
    assert [hit.document_id for hit in hits] == ["d1", "d2"]
    filtered = store.search([1.0, 0.0], top_k=2, document_ids=["d2"])
    assert [hit.document_id for hit in filtered] == ["d2"]
    assert filtered[0].text == "beta"


def test_soft_delete_does_not_hide_other_docs_and_rebuild() -> None:
    store = FaissVectorStore()
    store.add_document(_doc("d1"), ["keep-me-first"], [[1.0, 0.0]])
    store.add_document(_doc("d2"), ["delete-me"], [[0.9, 0.1]])
    store.add_document(_doc("d3"), ["keep-me-second"], [[0.8, 0.2]])
    assert store.delete_document("d2") is True
    hits = store.search([1.0, 0.0], top_k=2)
    assert [hit.document_id for hit in hits] == ["d1", "d3"]
    store.rebuild()
    rebuilt = store.search([1.0, 0.0], top_k=2)
    assert [hit.document_id for hit in rebuilt] == ["d1", "d3"]
    assert store.list_documents()
