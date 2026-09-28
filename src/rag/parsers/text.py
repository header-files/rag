"""Plain-text and Markdown parsers."""

from __future__ import annotations

from rag.errors import RagError
from rag.parsers.base import ParsedDocument


class TextParser:
    suffixes = (".txt", ".md", ".markdown")

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        return ParsedDocument(text=_decode(data, filename))


def _decode(data: bytes, filename: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise RagError(
        message=f"Unable to decode text file '{filename}'",
        type="unsupported_file",
    )
