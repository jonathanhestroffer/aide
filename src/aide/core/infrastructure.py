from pydantic import BaseModel, Field


class InfrastructureConfig(BaseModel):
    """Configuration for ML platform infrastructure concerns."""

    tracking_uri: str | None = Field(
        default=None, description="The URI for the tracking server (e.g., MLflow)."
    )

    artifact_uri: str | None = Field(
        default=None, description="The URI for the artifact storage (e.g., S3, GCS, local path)."
    )

    save_dir: str | None = Field(
        default=None, description="The directory to save model checkpoints and logs."
    )
