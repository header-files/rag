"""Parsed document payload."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
