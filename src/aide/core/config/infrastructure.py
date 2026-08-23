from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from aide.core.environment import get_project_path


def _is_remote_uri(value: str) -> bool:
    parsed = urlparse(value)
    if not parsed.scheme:
        return False
    # Local file or sqlite paths are handled locally
    return parsed.scheme not in {"file", "sqlite"}


def _resolve_sqlite_uri(value: str) -> str:
    raw = value[len("sqlite:") :]

    # Handle sqlite:////absolute/path vs sqlite:///relative/path
    if raw.startswith("////"):
        file_path = Path(raw[4:])
    elif raw.startswith("///"):
        file_path = Path(raw[3:])
    elif raw.startswith("//"):
        # In-memory or URI parameters (e.g., sqlite:///:memory:)
        return value
    else:
        file_path = Path(raw)

    if not file_path.is_absolute():
        file_path = get_project_path() / file_path

    absolute_path = file_path.resolve()
    return f"sqlite:////{absolute_path.as_posix().lstrip('/')}"


def _resolve_project_path(value: str | None) -> str | None:
    if value is None:
        return None

    if _is_remote_uri(value):
        return value

    parsed = urlparse(value)
    if parsed.scheme == "sqlite":
        return _resolve_sqlite_uri(value)

    # Strip file:// scheme if present for local paths
    path_str = value[7:] if value.startswith("file://") else value

    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    return str((get_project_path() / path).resolve())


class InfrastructureConfig(BaseModel):
    """Configuration for ML platform infrastructure concerns."""

    backend: str = Field(
        default="local",
        description="The backend to use for the ML platform infrastructure.",
    )

    tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="The URI for the tracking server (e.g., MLflow).",
    )

    artifact_location: str = Field(
        default="artifacts",
        description="The URI for the artifact storage (e.g., S3, GCS, local path).",
    )

    save_dir: str = Field(
        default="checkpoints",
        description="The directory to save model checkpoints and logs.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for field_name in ("artifact_location", "save_dir", "tracking_uri"):
            value = data.get(field_name)
            if isinstance(value, str):
                data[field_name] = _resolve_project_path(value)
        return data
