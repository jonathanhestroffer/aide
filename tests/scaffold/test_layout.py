from pathlib import PurePosixPath

import pytest

from aide.scaffold.layout import DEFAULT_LAYOUT, ScaffoldFile

EXPECTED_PATHS = {
    "configs/config.yaml",
    "configs/experiment/default.yaml",
    "configs/model/cnn.yaml",
    "configs/datamodule/artifact.yaml",
    "configs/trainer/default.yaml",
    "configs/infrastructure/local.yaml",
    "configs/checkpoint/default.yaml",
    ".env",
    "plugins/__init__.py",
    "plugins/models/__init__.py",
    "plugins/models/cnn.py",
    "plugins/components/__init__.py",
    "plugins/components/transforms.py",
}


def test_scaffold_file_is_immutable():
    scaffold_file = ScaffoldFile(
        relative_path="test.txt",
        template_name="test",
    )

    with pytest.raises(AttributeError):
        scaffold_file.relative_path = "other.txt"  # type: ignore


def test_default_layout_contains_expected_files():
    paths = {file.relative_path for file in DEFAULT_LAYOUT}

    assert paths == EXPECTED_PATHS


def test_default_layout_has_unique_paths():
    paths = [file.relative_path for file in DEFAULT_LAYOUT]

    assert len(paths) == len(set(paths))


def test_default_layout_has_unique_templates():
    templates = [file.template_name for file in DEFAULT_LAYOUT]

    assert len(templates) == len(set(templates))


def test_default_layout_paths_are_relative():
    for file in DEFAULT_LAYOUT:
        path = PurePosixPath(file.relative_path)

        assert not path.is_absolute()
        assert ".." not in path.parts


def test_default_layout_templates_are_names():
    for file in DEFAULT_LAYOUT:
        assert "/" not in file.template_name
        assert "\\" not in file.template_name
