"""Tests for generic env loader."""

from __future__ import annotations

from pathlib import Path

from common.config.env import load_env


def test_load_env_file_and_prefix_filter(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("FOO=1\nMODEL_X_NAME=n\n", encoding="utf-8")
    values = load_env(
        path,
        environ={"MODEL_X_API_KEY": "k", "OTHER": "nope"},
        prefixes=("MODEL_",),
        keys=("DEFAULT_MODEL",),
    )
    assert values["FOO"] == "1"
    assert values["MODEL_X_NAME"] == "n"
    assert values["MODEL_X_API_KEY"] == "k"
    assert "OTHER" not in values
