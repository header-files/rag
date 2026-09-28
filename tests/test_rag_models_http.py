"""Embedding / rerank HTTP adapter tests."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from common.config.endpoints import EndpointConfig
from llm.errors import LLMError
from rag.embedding.client import EmbeddingClient
from rag.rerank.client import RerankClient


@respx.mock
def test_embed_batches() -> None:
    route = respx.post("https://example.com/v1/embeddings").mock(
        side_effect=[
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]}),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 1]}]}),
        ]
    )
    client = EmbeddingClient(
        EndpointConfig(
            id="bge",
            protocol="openai-embeddings",
            name="bge",
            base_url="https://example.com/v1",
            api_key="sk",
            extra={"dimensions": "2"},
        ),
        batch_size=1,
    )
    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    body = json.loads(route.calls[0].request.content.decode())
    assert body["dimensions"] == 2


@respx.mock
def test_embed_error_and_cohere_rerank() -> None:
    respx.post("https://example.com/v1/embeddings").mock(
        return_value=httpx.Response(
            401, json={"error": {"message": "bad key", "code": "invalid_api_key"}}
        )
    )
    embed = EmbeddingClient(
        EndpointConfig(
            id="bge",
            protocol="openai-embeddings",
            name="bge",
            base_url="https://example.com/v1",
            api_key="sk",
        )
    )
    with pytest.raises(LLMError, match="bad key"):
        embed.embed(["x"])

    route = respx.post("https://example.com/v1/rerank").mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 1, "relevance_score": 0.9}]}
        )
    )
    rerank = RerankClient(
        EndpointConfig(
            id="r",
            protocol="cohere-rerank",
            name="rerank",
            base_url="https://example.com/v1",
            api_key="sk",
        )
    )
    results = rerank.rerank("q", ["a", "b"], top_n=1)
    assert results[0].index == 1
    assert json.loads(route.calls.last.request.content.decode())["top_n"] == 1


@respx.mock
def test_dashscope_text_rerank() -> None:
    route = respx.post(
        "https://example.cn-beijing.maas.aliyuncs.com/api/v1"
        "/services/rerank/text-rerank/text-rerank"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.93},
                        {"index": 2, "relevance_score": 0.34},
                    ]
                },
                "usage": {"prompt_tokens": 79, "total_tokens": 79},
            },
        )
    )
    client = RerankClient(
        EndpointConfig(
            id="qwen",
            protocol="dashscope-text-rerank",
            name="qwen3.7-text-rerank",
            base_url="https://example.cn-beijing.maas.aliyuncs.com/api/v1",
            api_key="sk-dash",
            extra={
                "instruct": (
                    "Given a web search query, retrieve relevant passages "
                    "that answer the query."
                )
            },
        )
    )
    results = client.rerank(
        "什么是文本排序模型",
        [
            "文本排序模型广泛用于搜索引擎",
            "量子计算是计算科学的一个前沿领域",
            "预训练语言模型的发展",
        ],
        top_n=2,
    )
    assert [(item.index, item.score) for item in results] == [(0, 0.93), (2, 0.34)]
    body = json.loads(route.calls.last.request.content.decode())
    assert body["model"] == "qwen3.7-text-rerank"
    assert body["input"]["query"] == "什么是文本排序模型"
    assert len(body["input"]["documents"]) == 3
    assert body["parameters"]["top_n"] == 2
    assert "instruct" in body["parameters"]
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-dash"
