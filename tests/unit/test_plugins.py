from __future__ import annotations

from pathlib import Path

import pytest

from aide.utils import plugins


def test_user_plugin_packages_are_loaded_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AIDE_PLUGINS=example_plugin, another_plugin\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_discover_and_import(*, user_packages=None, user_files=None, user_plugin_dirs=None):
        captured["user_packages"] = user_packages
        captured["user_files"] = user_files
        captured["user_plugin_dirs"] = user_plugin_dirs
        return {}

    monkeypatch.setattr(plugins, "discover_and_import", fake_discover_and_import)
    monkeypatch.delenv("AIDE_PLUGINS", raising=False)
    monkeypatch.delenv("ML_PLATFORM_PLUGINS", raising=False)

    plugins.load_plugins(env_file=env_file)

    assert captured["user_packages"] == ["example_plugin", "another_plugin"]
    assert (
        captured["user_plugin_dirs"] == [str((tmp_path / "plugins").resolve())]
        or captured["user_plugin_dirs"] is None
    )


def test_load_plugins_merges_explicit_and_env_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AIDE_PLUGINS=foo,bar\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_discover_and_import(*, user_packages=None, user_files=None, user_plugin_dirs=None):
        captured["user_packages"] = user_packages
        captured["user_plugin_dirs"] = user_plugin_dirs
        return {}

    monkeypatch.setattr(plugins, "discover_and_import", fake_discover_and_import)
    monkeypatch.delenv("AIDE_PLUGINS", raising=False)
    monkeypatch.delenv("ML_PLATFORM_PLUGINS", raising=False)

    plugins.load_plugins(user_plugins=["explicit"], env_file=env_file)

    assert captured["user_packages"] == ["explicit", "foo", "bar"]
