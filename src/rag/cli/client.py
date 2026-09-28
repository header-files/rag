"""HTTP client used by the RAG CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.logger import get_logger, new_trace_id

log = get_logger(__name__)


@dataclass
class ApiClientError(Exception):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message


class RagApiClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def upload(self, path: Path) -> dict:
        data = path.read_bytes()
        return self._request(
            "POST",
            "/documents",
            files={"file": (path.name, data)},
        )

    def list_documents(self) -> list[dict]:
        payload = self._request("GET", "/documents")
        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            raise ApiClientError(message="Invalid documents response")
        return documents

    def delete(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def create_session(self) -> str:
        payload = self._request("POST", "/sessions")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ApiClientError(message="Invalid session response")
        return session_id

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}")

    def ask(
        self,
        message: str,
        *,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        body: dict[str, object] = {"message": message, "stream": False}
        if session_id:
            body["session_id"] = session_id
        if document_ids:
            body["document_ids"] = document_ids
        return self._request("POST", "/chat", json=body)

    def ask_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[tuple[str, dict]]:
        body: dict[str, object] = {"message": message, "stream": True}
        if session_id:
            body["session_id"] = session_id
        if document_ids:
            body["document_ids"] = document_ids
        url = self.base_url + "/chat"
        headers = self._headers()
        log.info("api stream POST /chat")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, headers=headers, json=body) as response:
                    if response.status_code >= 400:
                        raw = response.read().decode("utf-8", errors="replace")
                        raise _error_from_body(response.status_code, raw)
                    for event, data in _iter_sse(response.iter_lines()):
                        try:
                            payload = json.loads(data) if data else {}
                        except json.JSONDecodeError as exc:
                            raise ApiClientError(message="Invalid SSE payload") from exc
                        if not isinstance(payload, dict):
                            raise ApiClientError(message="Invalid SSE payload")
                        yield event, payload
        except ApiClientError:
            raise
        except httpx.TimeoutException as exc:
            raise ApiClientError(message=str(exc) or "Request timed out") from exc
        except httpx.HTTPError as exc:
            raise ApiClientError(message=str(exc) or "HTTP error") from exc

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = self.base_url + path
        headers = self._headers()
        log.info("api %s %s", method, path)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiClientError(message=str(exc) or "Request timed out") from exc
        except httpx.HTTPError as exc:
            raise ApiClientError(message=str(exc) or "HTTP error") from exc
        if response.status_code >= 400:
            raise _error_from_body(response.status_code, response.text)
        if response.status_code == 204 or not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise ApiClientError(
                message="Invalid JSON response",
                status_code=response.status_code,
            )
        return payload

    def _headers(self) -> dict[str, str]:
        return {"X-Trace-Id": new_trace_id()}


def _error_from_body(status_code: int, body: str) -> ApiClientError:
    message = body.strip() or f"HTTP {status_code}"
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
        elif isinstance(error, str):
            message = error
    log.error("api error status=%s message=%s", status_code, message)
    return ApiClientError(message=message, status_code=status_code)


def _iter_sse(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)
