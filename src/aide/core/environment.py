from __future__ import annotations

import os
from pathlib import Path


def load_environment(env_file: str | Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from an env file into process environment."""
    env_path = _resolve_env_file(env_file)
    applied: dict[str, str] = {}
    if env_path is None or not env_path.is_file():
        return applied

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip()
        os.environ.setdefault(env_key, env_value)
        applied[env_key] = os.environ[env_key]

    return applied


def get_project_path(env_file: str | Path | None = None) -> Path:
    """Resolve project root from env var or env-file location."""
    env_path = _resolve_env_file(env_file)
    load_environment(env_path)

    configured = os.environ.get("AIDE_PROJECT_ROOT")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute() and env_path is not None:
            path = env_path.parent / path
        return path.resolve()

    if env_path is not None:
        return env_path.parent.resolve()

    return Path.cwd().resolve()


def get_config_path(env_file: str | Path | None = None) -> Path:
    """Resolve configs path from env variable or project fallback."""
    env_path = _resolve_env_file(env_file)
    root = get_project_path(env_path)
    configured = os.environ.get("AIDE_CONFIGS_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    return (root / "configs").resolve()


def get_plugins_path(env_file: str | Path | None = None) -> Path:
    """Resolve plugins path from env variable or project fallback."""
    env_path = _resolve_env_file(env_file)
    root = get_project_path(env_path)
    configured = os.environ.get("AIDE_PLUGINS_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    return (root / "plugins").resolve()


def _resolve_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is None:
        candidate = Path(".env").resolve()
        return candidate if candidate.is_file() else None

    return Path(env_file).expanduser().resolve()
