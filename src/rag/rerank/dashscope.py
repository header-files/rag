"""DashScope / 阿里云百炼 text-rerank adapter."""

from __future__ import annotations

from typing import Any

from llm.errors import LLMError
from llm.protocols.common import join_url, raise_from_http_body
from rag.rerank.base import RerankAdapter, RerankResult

_PATH = "/services/rerank/text-rerank/text-rerank"
_PROTOCOL = "dashscope-text-rerank"


class DashScopeTextRerankAdapter(RerankAdapter):
    """Aliyun Bailian / DashScope ``text-rerank`` HTTP shape.

    Request::

        {
          "model": "...",
          "input": {"query": "...", "documents": [...]},
          "parameters": {"top_n": N, "instruct": "..."}
        }

    Response::

        {"output": {"results": [{"index": 0, "relevance_score": 0.9}, ...]}}
    """

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        base = self.config.base_url
        if base.endswith("/text-rerank"):
            return base
        return join_url(base, _PATH)

    def _body(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for key, value in self.config.extra.items():
            if key == "top_n":
                try:
                    parameters["top_n"] = int(value)
                except ValueError as exc:
                    raise LLMError(
                        type="unsupported_parameter",
                        message=f"Invalid integer for top_n: {value}",
                    ) from exc
            else:
                parameters[key] = value
        if top_n is not None:
            parameters["top_n"] = top_n
        body: dict[str, Any] = {
            "model": self.config.name,
            "input": {"query": query, "documents": documents},
        }
        if parameters:
            body["parameters"] = parameters
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
                protocol=_PROTOCOL,
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMError(type="api_error", message="Invalid rerank response")
        output = payload.get("output")
        if not isinstance(output, dict):
            raise LLMError(
                type="api_error",
                message="Rerank response missing output",
            )
        rows = output.get("results")
        if not isinstance(rows, list):
            raise LLMError(
                type="api_error",
                message="Rerank response missing output.results",
            )
        results: list[RerankResult] = []
        for row in rows:
            if not isinstance(row, dict) or "index" not in row:
                raise LLMError(type="api_error", message="Rerank item missing index")
            score = row.get("relevance_score", row.get("score", 0.0))
            results.append(RerankResult(index=int(row["index"]), score=float(score)))
        return results
