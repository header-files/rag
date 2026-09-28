"""RAG HTTP routes."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.errors import error_response
from common.logger import get_logger
from llm.errors import LLMError
from rag.errors import RagError
from rag.runtime import RagRuntime

log = get_logger(__name__)

_STATUS = {
    "not_found": 404,
    "payload_too_large": 413,
    "unsupported_file": 400,
    "empty_document": 400,
}


class ChatBody(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    document_ids: list[str] | None = None
    stream: bool = False


def build_rag_router(app: FastAPI) -> APIRouter:
    router = APIRouter(tags=["rag"])

    def _runtime() -> RagRuntime:
        return app.state.runtime

    @router.get("/documents")
    def list_documents() -> dict[str, object]:
        return {
            "documents": [item.to_dict() for item in _runtime().store.list_documents()]
        }

    @router.get("/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, object]:
        document = _runtime().store.get_document(document_id)
        if document is None:
            raise RagError(
                message=f"Unknown document '{document_id}'",
                type="not_found",
            )
        return document.to_dict()

    @router.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> Response:
        deleted = _runtime().store.delete_document(document_id)
        if not deleted:
            raise RagError(
                message=f"Unknown document '{document_id}'",
                type="not_found",
            )
        log.info("Deleted document id=%s", document_id)
        return Response(status_code=204)

    @router.post("/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
        data = await file.read()
        document = _runtime().ingest.ingest_bytes(
            data,
            file.filename or "document",
            media_type=file.content_type,
        )
        return document.to_dict()

    @router.post("/sessions")
    def create_session() -> dict[str, str]:
        session = _runtime().sessions.create()
        return {"session_id": session.session_id}

    @router.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        deleted = _runtime().sessions.delete(session_id)
        if not deleted:
            raise RagError(
                message=f"Unknown session '{session_id}'",
                type="not_found",
            )
        return Response(status_code=204)

    @router.post("/chat")
    def chat(body: ChatBody):
        current = _runtime()
        prepared = current.query.prepare(
            body.message,
            session_id=body.session_id,
            document_ids=body.document_ids,
        )
        citations = [item.to_dict() for item in prepared.citations]
        if body.stream:
            return StreamingResponse(
                _event_stream(current, prepared, citations),
                media_type="text/event-stream",
            )
        answer = current.query.complete(prepared)
        return {
            "session_id": answer.session_id,
            "answer": answer.content,
            "citations": citations,
            "model": answer.model,
            "finish_reason": answer.finish_reason,
        }

    return router


def install_rag_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagError)
    async def _rag_error(_, exc: RagError):
        status = _STATUS.get(exc.type, 400)
        log.error("RAG error type=%s message=%s", exc.type, exc.message)
        return error_response(
            status_code=status,
            error_type=exc.type,
            message=exc.message,
        )


def _event_stream(runtime: RagRuntime, prepared, citations: list[dict]) -> Iterator[str]:
    try:
        yield _sse("session", {"session_id": prepared.session_id})
        yield _sse("citations", {"citations": citations})
        for chunk in runtime.query.iter_stream(prepared):
            if chunk.type == "content" and chunk.delta:
                yield _sse("content", {"delta": chunk.delta})
            elif chunk.type == "finish":
                yield _sse(
                    "done",
                    {"finish_reason": chunk.finish_reason, "model": chunk.model},
                )
    except (RagError, LLMError) as exc:
        error_type = getattr(exc, "type", "rag_error")
        log.error("chat stream failed type=%s message=%s", error_type, exc)
        yield _sse("error", {"type": error_type, "message": str(exc)})


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
