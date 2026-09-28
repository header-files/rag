"""Public FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from api.errors import register_common_exception_handlers
from api.middleware import install_trace_middleware
from common.logger import setup_logging
from rag.api import build_rag_router, install_rag_exception_handlers
from rag.runtime import RagRuntime


def create_app(
    runtime: RagRuntime | None = None,
    *,
    env_path: str | Path | None = None,
) -> FastAPI:
    if runtime is None:
        setup_logging(env_path=env_path)
        runtime = RagRuntime.from_env(env_path)

    app = FastAPI(title="rag", version="0.1.0")
    app.state.runtime = runtime
    install_trace_middleware(app)
    register_common_exception_handlers(app)
    install_rag_exception_handlers(app)

    @app.get("/health")
    def health() -> dict[str, object]:
        current: RagRuntime = app.state.runtime
        return {
            "status": "ok",
            "vector_store": current.settings.vector_store,
            "documents": len(current.store.list_documents()),
        }

    app.include_router(build_rag_router(app))
    return app
