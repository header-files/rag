"""Generic .env loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def load_env(
    env_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    prefixes: tuple[str, ...] | None = None,
    keys: tuple[str, ...] | None = None,
) -> dict[str, str | None]:
    """Load values from a .env file and optionally merge process env.

    Args:
        env_path: Path to .env. Defaults to ``./.env``.
        environ: Explicit env mapping. ``None`` means ``os.environ``;
            pass ``{}`` to ignore process env and only use the file.
        prefixes: If set, only process-env keys with these prefixes are merged.
        keys: If set, also merge these exact process-env keys.

    Returns:
        Merged key/value map (values may be None from dotenv).
    """
    values: dict[str, str | None] = {}

    path = Path(env_path) if env_path else Path.cwd() / ".env"
    if path.is_file():
        values.update(dotenv_values(path))

    source = environ if environ is not None else os.environ
    for key, value in source.items():
        if prefixes is not None or keys is not None:
            matched = False
            if prefixes and any(key.startswith(p) for p in prefixes):
                matched = True
            if keys and key in keys:
                matched = True
            if not matched:
                continue
        values[key] = value

    return values
