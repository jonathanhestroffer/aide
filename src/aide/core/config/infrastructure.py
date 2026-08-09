from __future__ import annotations

from pydantic import BaseModel, Field

from aide.core.config.checkpoint import CheckpointConfig


class InfrastructureConfig(BaseModel):
    """Configuration for ML platform infrastructure concerns."""

    backend: str = Field(
        default="local", description="The backend to use for the ML platform infrastructure."
    )

    tracking_uri: str | None = Field(
        default=None, description="The URI for the tracking server (e.g., MLflow)."
    )

    artifact_location: str | None = Field(
        default=None, description="The URI for the artifact storage (e.g., S3, GCS, local path)."
    )

    plugins: list[str] = Field(
        default_factory=list,
        description="List of plugins to load for the ML platform infrastructure.",
    )

    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="The checkpoint configuration."
    )

    save_dir: str | None = Field(
        default=None, description="The directory to save model checkpoints and logs."
    )
