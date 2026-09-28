"""Cohere-compatible rerank adapter."""

from __future__ import annotations

from typing import Any

from common.config.endpoints import extra_json_fields
from llm.errors import LLMError
from llm.protocols.common import join_url, raise_from_http_body
from rag.rerank.base import RerankAdapter, RerankResult


class CohereRerankAdapter(RerankAdapter):
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url
        if base.endswith("/rerank"):
            return base
        return join_url(base, "/rerank")

    def _body(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = extra_json_fields(self.config.extra)
        body["model"] = self.config.name
        body["query"] = query
        body["documents"] = documents
        if top_n is not None:
            body["top_n"] = top_n
        return body

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        response = self.transport.request(
            "POST",
            self._url(),
            headers=self._headers(),
            json=self._body(query, documents, top_n=top_n),
        )
        if response.status_code >= 400:
            raise_from_http_body(
                response.status_code,
                response.text,
                protocol="cohere-rerank",
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMError(type="api_error", message="Invalid rerank response")
        rows = payload.get("results")
        if rows is None:
            rows = payload.get("data")
        if not isinstance(rows, list):
            raise LLMError(type="api_error", message="Rerank response missing results")
        results: list[RerankResult] = []
        for row in rows:
            if not isinstance(row, dict) or "index" not in row:
                raise LLMError(type="api_error", message="Rerank item missing index")
            score = row.get("relevance_score", row.get("score", 0.0))
            results.append(RerankResult(index=int(row["index"]), score=float(score)))
        return results
