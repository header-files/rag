"""Extensible logging context (trace_id and future fields).

Context is stored in a ``contextvars.ContextVar`` as a string dict so async
and threaded call paths stay isolated. Bind arbitrary keys for later use in
``LOG_FORMAT`` (e.g. ``%(user_id)s``); ensure missing standard fields via
``LOG_ENSURE_FIELDS`` / ``ContextFilter``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_log_context: ContextVar[dict[str, str]] = ContextVar("log_context", default={})


def get_context() -> dict[str, str]:
    """Return a shallow copy of the current log context."""
    return dict(_log_context.get())


def bind(**fields: Any) -> Token[dict[str, str]]:
    """Merge fields into the current context; returns a token for reset."""
    merged = get_context()
    for key, value in fields.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = str(value)
    return _log_context.set(merged)


def reset(token: Token[dict[str, str]]) -> None:
    """Restore context to the state captured by ``bind`` / ``clear`` token."""
    _log_context.reset(token)


def clear() -> Token[dict[str, str]]:
    """Clear all context fields; returns a token for reset."""
    return _log_context.set({})


def new_trace_id() -> str:
    """Generate a new opaque trace id (UUID4 hex)."""
    return uuid.uuid4().hex


def get_trace_id() -> str | None:
    return get_context().get("trace_id")


def set_trace_id(trace_id: str | None) -> Token[dict[str, str]]:
    """Set or clear ``trace_id`` in the current context."""
    return bind(trace_id=trace_id)


@contextmanager
def log_context(**fields: Any) -> Iterator[Mapping[str, str]]:
    """Temporarily bind fields for the duration of the ``with`` block."""
    token = bind(**fields)
    try:
        yield get_context()
    finally:
        reset(token)


@contextmanager
def traced(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id (generated if omitted) for the ``with`` block."""
    value = trace_id or new_trace_id()
    with log_context(trace_id=value):
        yield value
