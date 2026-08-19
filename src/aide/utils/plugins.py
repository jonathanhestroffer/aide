from __future__ import annotations

import os
from pathlib import Path

from aide.core.environment import (
    get_config_path,
    get_plugins_path,
    get_project_path,
    load_environment,
)
from aide.registry.discovery import discover_and_import


def load_env_file(env_file: str | Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from an env file and return applied keys."""
    return load_environment(env_file)


def project_root_from_env(env_file: str | Path | None = None) -> Path:
    """Resolve project root from env file and ML_PLATFORM_PROJECT_ROOT."""
    return get_project_path(env_file)


def configs_path_from_env(env_file: str | Path | None = None) -> Path:
    """Resolve configs path from env or project-root fallback."""
    return get_config_path(env_file)


def plugins_path_from_env(env_file: str | Path | None = None) -> Path:
    """Resolve plugins path from env or project-root fallback."""
    return get_plugins_path(env_file)


def ensure_project_env(env_file: str | Path | None = None) -> dict[str, str]:
    """Ensure project/config/plugin env vars are set and return resolved values."""
    if env_file is not None:
        load_env_file(env_file)

    root = project_root_from_env(env_file)
    configs = configs_path_from_env(env_file)
    plugins = plugins_path_from_env(env_file)

    os.environ.setdefault("ML_PLATFORM_PROJECT_ROOT", str(root))
    os.environ.setdefault("ML_PLATFORM_CONFIGS_PATH", str(configs))
    os.environ.setdefault("ML_PLATFORM_PLUGINS_PATH", str(plugins))

    return {
        "ML_PLATFORM_PROJECT_ROOT": os.environ["ML_PLATFORM_PROJECT_ROOT"],
        "ML_PLATFORM_CONFIGS_PATH": os.environ["ML_PLATFORM_CONFIGS_PATH"],
        "ML_PLATFORM_PLUGINS_PATH": os.environ["ML_PLATFORM_PLUGINS_PATH"],
    }


def _user_plugin_packages_from_env(env_file: str | Path | None = None) -> list[str] | None:
    """Extract additional plugin package names from environment state."""
    load_environment(env_file)

    raw_plugins = os.environ.get("AIDE_PLUGINS") or os.environ.get("ML_PLATFORM_PLUGINS")
    if not raw_plugins:
        return None

    packages = [plugin.strip() for plugin in raw_plugins.split(",") if plugin.strip()]
    return packages or None


def load_plugins(
    user_plugins: list[str] | None = None,
    *,
    env_file: str | Path | None = None,
    project_root: str | Path | None = None,
) -> None:
    """Load and register framework plus user plugins via discover_and_import."""
    env_packages = _user_plugin_packages_from_env(env_file)
    package_names = []

    if user_plugins:
        package_names.extend(user_plugins)
    if env_packages:
        package_names.extend(env_packages)

    package_names = list(dict.fromkeys(package_names)) if package_names else None

    if project_root is not None:
        plugin_dir = Path(project_root).expanduser().resolve() / "plugins"
    else:
        plugin_dir = plugins_path_from_env(env_file)

    local_dirs = [str(plugin_dir)] if plugin_dir.is_dir() else None
    discover_and_import(user_packages=package_names, user_plugin_dirs=local_dirs)


def load_plugins_from_env_path(env_file: str | Path | None = None) -> None:
    """Load plugin modules discovered from the env-derived project root."""
    load_plugins(None, env_file=env_file)
