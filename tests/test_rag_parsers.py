"""Parser tests."""

from __future__ import annotations

import io

import pytest
from docx import Document

from rag.errors import RagError
from rag.parsers import parse_document


def test_txt_md_docx() -> None:
    assert parse_document("你好".encode("gb18030"), "a.txt").text == "你好"
    assert "# t" in parse_document(b"# t\n\nbody", "a.md").text
    document = Document()
    document.add_paragraph("Hello DOCX")
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Hello DOCX" in parse_document(buffer.getvalue(), "a.docx").text


def test_unsupported_and_bad_docx() -> None:
    with pytest.raises(RagError, match="Unsupported") as exc:
        parse_document(b"x", "a.exe")
    assert exc.value.type == "unsupported_file"
    with pytest.raises(RagError, match="docx"):
        parse_document(b"not-a-docx", "a.docx")
