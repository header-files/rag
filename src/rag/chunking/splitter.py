"""Recursive text splitter."""

from __future__ import annotations


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...],
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    units = _split_units(stripped, chunk_size, separators)
    packed = _pack(units, chunk_size, chunk_overlap)
    return [chunk.strip() for chunk in packed if chunk.strip()]


def _split_units(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...],
) -> list[str]:
    if text == "":
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = ""
    rest: tuple[str, ...] = ("",)
    found = False
    for index, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            rest = separators[index + 1 :] or ("",)
            found = True
            break
        if candidate in text:
            separator = candidate
            rest = separators[index + 1 :] or ("",)
            found = True
            break
    if not found or separator == "":
        return [
            text[index : index + chunk_size]
            for index in range(0, len(text), chunk_size)
        ]

    parts = text.split(separator)
    units: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        piece = part if index == last else part + separator
        if piece == "":
            continue
        if len(piece) <= chunk_size:
            units.append(piece)
        else:
            units.extend(_split_units(piece, chunk_size, rest))
    return units


def _pack(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        if len(current) + len(unit) <= chunk_size:
            current += unit
            continue
        chunks.append(current)
        if chunk_overlap > 0:
            tail = current[-chunk_overlap:]
            joined = tail + unit
            current = joined if len(joined) <= chunk_size else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks
