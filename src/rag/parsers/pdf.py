"""PDF parser."""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.errors import RagError
from rag.parsers.base import ParsedDocument


class PdfParser:
    suffixes = (".pdf",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(data))
        except PdfReadError as exc:
            raise RagError(
                message=f"Failed to parse pdf '{filename}': {exc}",
                type="unsupported_file",
            ) from exc

        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return ParsedDocument(text="\n\n".join(pages))
