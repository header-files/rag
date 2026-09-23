"""Tests for flat .env multi-model config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config.errors import ConfigError
from common.config.models import get_model, load_models


def test_load_models_from_environ() -> None:
    env = {
        "DEFAULT_MODEL": "chatgpt",
        "MODEL_CHATGPT_PROTOCOL": "openai-completions",
        "MODEL_CHATGPT_NAME": "gpt-4o-mini",
        "MODEL_CHATGPT_BASE_URL": "https://api.openai.com/v1",
        "MODEL_CHATGPT_API_KEY": "sk-test",
        "MODEL_CLAUDE_PROTOCOL": "anthropic-messages",
        "MODEL_CLAUDE_NAME": "claude-3-5-sonnet-latest",
        "MODEL_CLAUDE_BASE_URL": "https://api.anthropic.com",
        "MODEL_CLAUDE_API_KEY": "sk-ant-test",
    }
    default_id, models = load_models(environ=env)
    assert default_id == "chatgpt"
    assert set(models) == {"chatgpt", "claude"}
    assert models["chatgpt"].protocol == "openai-completions"
    assert models["claude"].protocol == "anthropic-messages"
    assert models["chatgpt"].base_url == "https://api.openai.com/v1"


def test_protocol_defaults_to_openai_completions() -> None:
    env = {
        "MODEL_FOO_NAME": "m",
        "MODEL_FOO_BASE_URL": "https://example.com/v1",
        "MODEL_FOO_API_KEY": "k",
    }
    _, models = load_models(environ=env)
    assert models["foo"].protocol == "openai-completions"


def test_missing_required_field() -> None:
    env = {
        "MODEL_FOO_NAME": "m",
        "MODEL_FOO_BASE_URL": "https://example.com/v1",
    }
    with pytest.raises(ConfigError, match="missing required fields"):
        load_models(environ=env)


def test_unknown_model_id() -> None:
    env = {
        "MODEL_FOO_NAME": "m",
        "MODEL_FOO_BASE_URL": "https://example.com/v1",
        "MODEL_FOO_API_KEY": "k",
    }
    with pytest.raises(ConfigError, match="Unknown model"):
        get_model("bar", environ=env)


def test_load_from_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "DEFAULT_MODEL=demo",
                "MODEL_DEMO_PROTOCOL=openai-responses",
                "MODEL_DEMO_NAME=gpt",
                "MODEL_DEMO_BASE_URL=https://example.com/v1",
                "MODEL_DEMO_API_KEY=secret",
            ]
        ),
        encoding="utf-8",
    )
    default_id, models = load_models(path, environ={})
    assert default_id == "demo"
    assert models["demo"].protocol == "openai-responses"
    assert models["demo"].api_key == "secret"
