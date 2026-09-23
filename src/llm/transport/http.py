"""HTTP transport helpers built on httpx."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx

from llm.errors import LLMError
from common.logger import get_logger

log = get_logger(__name__)


def _map_http_error(exc: Exception) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMError(
            type="timeout_error",
            message=str(exc) or "Request timed out",
            retryable=True,
        )
    if isinstance(exc, httpx.NetworkError):
        return LLMError(
            type="network_error",
            message=str(exc) or "Network error",
            retryable=True,
        )
    return LLMError(
        type="network_error",
        message=str(exc) or "HTTP transport error",
        retryable=None,
    )


class HttpTransport:
    def __init__(self, *, timeout: float = 60.0) -> None:
        self._timeout = timeout

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            log.debug("HTTP %s %s", method, url)
            with httpx.Client(timeout=self._timeout) as client:
                response = client.request(
                    method, url, headers=dict(headers), json=json
                )
            log.debug("HTTP %s %s -> %s", method, url, response.status_code)
            return response
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_http_error(exc) from exc

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        try:
            log.debug("HTTP stream %s %s", method, url)
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    method, url, headers=dict(headers), json=json
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        yield f"__HTTP_ERROR__:{response.status_code}:{body}"
                        return
                    for line in response.iter_lines():
                        yield line
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_http_error(exc) from exc

    async def arequest(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            log.debug("HTTP async %s %s", method, url)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(
                    method, url, headers=dict(headers), json=json
                )
            log.debug("HTTP async %s %s -> %s", method, url, response.status_code)
            return response
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_http_error(exc) from exc

    async def astream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        try:
            log.debug("HTTP async stream %s %s", method, url)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    method, url, headers=dict(headers), json=json
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(
                            "utf-8", errors="replace"
                        )
                        yield f"__HTTP_ERROR__:{response.status_code}:{body}"
                        return
                    async for line in response.aiter_lines():
                        yield line
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_http_error(exc) from exc
