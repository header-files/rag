"""Configuration-related errors (shared across the project)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""

    message: str

    def __str__(self) -> str:
        return self.message
