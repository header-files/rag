"""Logging filters that inject context fields onto LogRecord."""

from __future__ import annotations

import logging

from common.logger.context import get_context


class ContextFilter(logging.Filter):
    """Attach context values to each record for use in format strings.

    - All current context keys are set as attributes on the record.
    - ``ensure_fields`` are always present (default ``"-"`` when unbound),
      so formatters like ``%(trace_id)s`` never raise KeyError.
    - Future fields: ``bind(user_id=...)`` and add ``%(user_id)s`` to
      ``LOG_FORMAT``, optionally listing ``user_id`` in ``ensure_fields``.
    """

    def __init__(
        self,
        name: str = "",
        *,
        ensure_fields: tuple[str, ...] = ("trace_id",),
        default: str = "-",
    ) -> None:
        super().__init__(name)
        self.ensure_fields = ensure_fields
        self.default = default

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_context()
        for key, value in ctx.items():
            setattr(record, key, value)
        for key in self.ensure_fields:
            if not hasattr(record, key):
                setattr(record, key, self.default)
        return True
