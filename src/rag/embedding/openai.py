"""OpenAI-compatible embeddings adapter."""

from __future__ import annotations

from typing import Any

from common.config.endpoints import extra_json_fields
from llm.errors import LLMError
from llm.protocols.common import join_url, raise_from_http_body
from rag.embedding.base import EmbeddingAdapter


class OpenAIEmbeddingsAdapter(EmbeddingAdapter):
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url
        if base.endswith("/embeddings"):
            return base
        return join_url(base, "/embeddings")

    def _body(self, texts: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = extra_json_fields(self.config.extra)
        body["model"] = self.config.name
        body["input"] = texts
        return body

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.transport.request(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._body(texts),
        )
        if response.status_code >= 400:
            raise_from_http_body(
                response.status_code,
                response.text,
                protocol="openai-embeddings",
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMError(type="api_error", message="Invalid embeddings response")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise LLMError(type="api_error", message="Embeddings response missing data")
        ordered = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: int(row.get("index") or 0),
        )
        vectors: list[list[float]] = []
        for row in ordered:
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise LLMError(
                    type="api_error", message="Embedding item is not a vector"
                )
            vectors.append([float(value) for value in embedding])
        if len(vectors) != len(texts):
            raise LLMError(
                type="api_error",
                message=(
                    f"Embedding count mismatch: expected {len(texts)}, "
                    f"got {len(vectors)}"
                ),
            )
        return vectors
