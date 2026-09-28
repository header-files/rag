"""Tests for embedding, rerank, API, and RAG settings."""

from __future__ import annotations

import pytest

from common.config.api import load_api_settings
from common.config.embeddings import get_embedding, load_embeddings
from common.config.errors import ConfigError
from common.config.rag import load_rag_settings
from common.config.rerank import load_rerankers


def _embedding_env() -> dict[str, str]:
    return {
        "DEFAULT_EMBEDDING": "bge",
        "EMBEDDING_BGE_NAME": "bge-m3",
        "EMBEDDING_BGE_BASE_URL": "https://example.com/v1",
        "EMBEDDING_BGE_API_KEY": "sk-embed",
        "EMBEDDING_BGE_DIMENSIONS": "1024",
    }


def test_load_embeddings() -> None:
    default_id, models = load_embeddings(environ=_embedding_env())
    assert default_id == "bge"
    assert models["bge"].extra["dimensions"] == "1024"
    assert get_embedding(environ=_embedding_env()).id == "bge"


def test_load_rerankers() -> None:
    env = {
        "RERANK_COHERE_PROTOCOL": "cohere-rerank",
        "RERANK_COHERE_NAME": "rerank-v3.5",
        "RERANK_COHERE_BASE_URL": "https://api.cohere.com/v1",
        "RERANK_COHERE_API_KEY": "sk-rerank",
    }
    default_id, models = load_rerankers(environ=env)
    assert default_id == "cohere"
    assert models["cohere"].protocol == "cohere-rerank"


def test_rag_settings_validates_rerank_le_retrieve() -> None:
    with pytest.raises(ConfigError, match="RAG_RERANK_TOP_N"):
        load_rag_settings(
            environ={"RAG_RETRIEVE_TOP_K": "5", "RAG_RERANK_TOP_N": "10"}
        )


def test_rag_settings_defaults() -> None:
    settings = load_rag_settings(
        environ={
            "RAG_CHUNK_SIZE": "200",
            "RAG_CHUNK_OVERLAP": "20",
            "RAG_RERANK_ENABLED": "true",
            "RAG_MAX_HISTORY_TURNS": "3",
        }
    )
    assert settings.vector_store == "faiss"
    assert settings.rerank_enabled is True
    assert settings.query_rewrite_enabled is False
    assert settings.max_history_turns == 3
    assert settings.chunk_size == 200


def test_query_rewrite_can_enable() -> None:
    settings = load_rag_settings(
        environ={
            "RAG_QUERY_REWRITE_ENABLED": "true",
            "RAG_QUERY_REWRITE_MODEL": "chatgpt",
        }
    )
    assert settings.query_rewrite_enabled is True
    assert settings.query_rewrite_model_id == "chatgpt"


def test_only_faiss_vector_store() -> None:
    with pytest.raises(ConfigError, match="Unsupported vector store"):
        load_rag_settings(environ={"RAG_VECTOR_STORE": "memory"})


def test_api_settings() -> None:
    settings = load_api_settings(
        environ={"API_HOST": "0.0.0.0", "API_PORT": "9000", "API_BASE_URL": "http://x/"}
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.base_url == "http://x"
