"""Chunker tests."""

from __future__ import annotations

from rag.chunking import split_text


def test_empty_and_short() -> None:
    assert split_text(" \n", chunk_size=10, chunk_overlap=0, separators=("\n", "")) == []
    assert split_text("hello", chunk_size=10, chunk_overlap=0, separators=("\n", "")) == [
        "hello"
    ]


def test_paragraph_split_and_overlap() -> None:
    chunks = split_text(
        "aaaa\n\nbbbb\n\ncccc",
        chunk_size=6,
        chunk_overlap=0,
        separators=("\n\n", "\n", ""),
    )
    assert chunks == ["aaaa", "bbbb", "cccc"]
    overlapped = split_text(
        "abcdefghij",
        chunk_size=4,
        chunk_overlap=2,
        separators=("",),
    )
    assert overlapped == ["abcd", "efgh", "ghij"]
