from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from aide.core.config.component import ComponentConfig


class DataModuleConfig(BaseModel):
    """Configuration for the PyTorch Lightning DataModule."""

    artifact_uri: str = Field(
        default="data", description="The URI for the data artifacts (e.g., S3, GCS, local path)."
    )

    transforms: list[ComponentConfig] | None = Field(
        default_factory=list, description="List of transform component configurations."
    )

    batch_size: int = Field(default=32, description="The batch size for the DataLoader.")

    num_workers: int | None = Field(
        default=None,
        description=(
            "The number of worker processes for the DataLoader. "
            "If omitted, defaults to the number of CPUs on the machine."
        ),
    )

    pin_memory: bool = Field(default=True, description="Whether to pin memory in the DataLoader.")

    persistent_workers: bool = Field(
        default=True, description="Whether to use persistent workers in the DataLoader."
    )

    @field_validator("transforms", mode="before")
    @classmethod
    def _normalize_transforms(
        cls,
        transforms: list[ComponentConfig] | None,
    ) -> list:
        if transforms is None:
            return []
        return transforms
