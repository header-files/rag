"""HTTP middleware: bind trace_id for logging."""

from __future__ import annotations

from fastapi import Request

from common.logger import bind, get_logger, new_trace_id, reset

log = get_logger(__name__)


def install_trace_middleware(app) -> None:
    @app.middleware("http")
    async def _trace(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or new_trace_id()
        token = bind(trace_id=trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            log.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        finally:
            reset(token)
