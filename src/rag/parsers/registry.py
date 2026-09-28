"""File-type parser registry."""

from __future__ import annotations

from pathlib import Path

from rag.errors import RagError
from rag.parsers.base import ParsedDocument
from rag.parsers.docx import DocxParser
from rag.parsers.pdf import PdfParser
from rag.parsers.text import TextParser

_PARSERS = (TextParser(), DocxParser(), PdfParser())
_BY_SUFFIX = {
    suffix: parser for parser in _PARSERS for suffix in parser.suffixes
}


def supported_suffixes() -> tuple[str, ...]:
    return tuple(sorted(_BY_SUFFIX))


def parse_document(data: bytes, filename: str) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    parser = _BY_SUFFIX.get(suffix)
    if parser is None:
        supported = ", ".join(supported_suffixes())
        label = suffix or filename
        raise RagError(
            message=f"Unsupported file type '{label}'. Supported: {supported}",
            type="unsupported_file",
        )
    return parser.parse(data, filename)
