"""Shared HTTP error payload helpers."""

from __future__ import annotations

from fastapi.responses import JSONResponse

from common.config.errors import ConfigError
from llm.errors import LLMError


def error_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
    extras: dict | None = None,
) -> JSONResponse:
    payload: dict[str, object] = {"type": error_type, "message": message}
    if extras:
        payload.update(extras)
    return JSONResponse(status_code=status_code, content={"error": payload})


def register_common_exception_handlers(app) -> None:
    @app.exception_handler(ConfigError)
    async def _config_error(_, exc: ConfigError) -> JSONResponse:
        return error_response(
            status_code=500,
            error_type="config_error",
            message=exc.message,
        )

    @app.exception_handler(LLMError)
    async def _llm_error(_, exc: LLMError) -> JSONResponse:
        return error_response(
            status_code=502,
            error_type=exc.type,
            message=exc.message,
            extras={
                "status_code": exc.status_code,
                "provider_code": exc.provider_code,
            },
        )
