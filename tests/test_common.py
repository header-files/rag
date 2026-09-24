"""Tests for shared protocol helpers."""

from __future__ import annotations

import pytest

from llm.errors import LLMError
from llm.protocols.common import (
    map_finish_reason,
    parse_usage_openai_chat,
    parse_usage_tokens,
    raise_from_http_body,
)


def test_map_finish_reason() -> None:
    assert map_finish_reason("stop") == "stop"
    assert map_finish_reason("end_turn") == "stop"
    assert map_finish_reason("max_tokens") == "length"
    assert map_finish_reason("content_filter") == "content_filter"
    assert map_finish_reason(None) is None
    assert map_finish_reason("unknown_xyz") is None


def test_parse_usage() -> None:
    u = parse_usage_openai_chat(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    )
    assert u is not None
    assert u.input_tokens == 1
    assert u.output_tokens == 2
    assert u.total_tokens == 3

    u2 = parse_usage_tokens({"input_tokens": 4, "output_tokens": 5})
    assert u2 is not None
    assert u2.total_tokens == 9


def test_raise_from_http_body_openai_style(caplog) -> None:
    body = '{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}'
    with caplog.at_level("ERROR"):
        with pytest.raises(LLMError) as exc:
            raise_from_http_body(401, body, protocol="openai-completions")
    err = exc.value
    assert err.type == "auth_error"
    assert err.message == "bad key"
    assert err.status_code == 401
    assert err.provider_code == "invalid_api_key"
    assert err.retryable is False
    assert "bad key" in caplog.text
    assert "openai-completions" in caplog.text


def test_raise_from_http_body_retryable_429() -> None:
    with pytest.raises(LLMError) as exc:
        raise_from_http_body(429, '{"error":{"message":"rate limited"}}', protocol="x")
    assert exc.value.retryable is True
    assert exc.value.type == "api_error"
