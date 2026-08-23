import os
from pathlib import Path

from aide.core.config.infrastructure import InfrastructureConfig


def test_save_dir_resolves_relative_path_to_project_root(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    original_cwd = Path.cwd()

    try:
        os.chdir(project_root)
        config = InfrastructureConfig(save_dir="workspace/checkpoints")

        assert Path(config.save_dir).is_absolute()
        assert Path(config.save_dir) == (project_root / "workspace" / "checkpoints").resolve()
    finally:
        os.chdir(original_cwd)


def test_artifact_location_preserves_remote_uri():
    config = InfrastructureConfig(artifact_location="s3://my-bucket/path")

    assert config.artifact_location == "s3://my-bucket/path"


def test_tracking_uri_resolves_relative_path_to_project_root(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    original_cwd = Path.cwd()

    try:
        os.chdir(project_root)
        config = InfrastructureConfig(tracking_uri="workspace/mlflow")

        assert Path(config.tracking_uri).is_absolute()
        assert Path(config.tracking_uri) == (project_root / "workspace" / "mlflow").resolve()
    finally:
        os.chdir(original_cwd)


def test_sqlite_database_uri_resolves_relative_path_to_project_root(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    original_cwd = Path.cwd()

    try:
        os.chdir(project_root)
        config = InfrastructureConfig(tracking_uri="sqlite:///./workspace/mlflow.db")

        assert config.tracking_uri.startswith("sqlite:////")
        assert Path(config.tracking_uri[len("sqlite:") :]).exists() is False
        assert Path(config.tracking_uri[len("sqlite:") :]).is_absolute()
        assert (
            Path(config.tracking_uri[len("sqlite:") :])
            == (project_root / "workspace" / "mlflow.db").resolve()
        )
    finally:
        os.chdir(original_cwd)
