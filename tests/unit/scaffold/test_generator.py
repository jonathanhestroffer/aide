from pathlib import Path

import pytest

from aide.scaffold.generator import (
    scaffold_experiment,
    write_dataset_manifest_uri,
)
from aide.scaffold.layout import DEFAULT_LAYOUT


def test_scaffold_experiment_creates_project(tmp_path: Path):
    result = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    expected_base = tmp_path

    assert result.base_path == expected_base
    assert expected_base.is_dir()

    assert set(result.created) == {file.relative_path for file in DEFAULT_LAYOUT}
    assert result.overwritten == ()
    assert result.skipped == ()

    for file_spec in DEFAULT_LAYOUT:
        path = expected_base / file_spec.relative_path

        assert path.is_file()


def test_scaffold_experiment_creates_nested_directories(tmp_path: Path):
    result = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    base_path = result.base_path

    assert (base_path / "configs").is_dir()
    assert (base_path / "configs" / "experiment").is_dir()
    assert (base_path / "configs" / "model").is_dir()
    assert (base_path / "configs" / "datamodule").is_dir()
    assert (base_path / "configs" / "trainer").is_dir()
    assert (base_path / "configs" / "infrastructure").is_dir()
    assert (base_path / "configs" / "checkpoint").is_dir()
    assert (base_path / "plugins").is_dir()
    assert (base_path / "plugins" / "models").is_dir()
    assert (base_path / "plugins" / "components").is_dir()


def test_scaffold_experiment_second_run_skips_existing_files(tmp_path: Path):
    first = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    second = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    assert set(first.created) == {file.relative_path for file in DEFAULT_LAYOUT}

    assert second.created == ()
    assert second.overwritten == ()
    assert set(second.skipped) == {file.relative_path for file in DEFAULT_LAYOUT}


def test_scaffold_experiment_overwrites_existing_files(tmp_path: Path):
    _ = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    config_path = tmp_path / "configs" / "experiment" / "default.yaml"

    original_contents = config_path.read_text(encoding="utf-8")

    config_path.write_text(
        "modified by test\n",
        encoding="utf-8",
    )

    second = scaffold_experiment(
        target_dir=str(tmp_path),
        overwrite=True,
    )

    assert second.created == ()
    assert second.skipped == ()
    assert set(second.overwritten) == {file.relative_path for file in DEFAULT_LAYOUT}

    restored_contents = config_path.read_text(encoding="utf-8")

    assert restored_contents == original_contents


def test_scaffold_experiment_preserves_unmanaged_files(tmp_path: Path):
    first = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    unmanaged_file = first.base_path / "user_file.txt"
    unmanaged_file.write_text(
        "do not delete me",
        encoding="utf-8",
    )

    scaffold_experiment(
        target_dir=str(tmp_path),
        overwrite=True,
    )

    assert unmanaged_file.exists()
    assert unmanaged_file.read_text(encoding="utf-8") == "do not delete me"


def test_scaffold_result_is_immutable(tmp_path: Path):
    result = scaffold_experiment(
        target_dir=str(tmp_path),
    )

    with pytest.raises(AttributeError):
        result.base_path = tmp_path  # type: ignore


def test_write_dataset_manifest_uri(tmp_path: Path):
    scaffold_experiment(
        target_dir=str(tmp_path),
    )

    base_path = tmp_path

    write_dataset_manifest_uri(
        base_path,
        "/shared/datasets/procedural_shapes/manifest.json",
    )

    env_path = base_path / ".env"
    contents = env_path.read_text(encoding="utf-8")

    assert ("AIDE_DATASET_MANIFEST=/shared/datasets/procedural_shapes/manifest.json") in contents
