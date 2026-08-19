from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from aide.core.config.component import ComponentConfig
from aide.core.config.dataloader import DataLoaderConfig


class DataModuleConfig(BaseModel):
    """Configuration for the PyTorch Lightning DataModule."""

    train_dataset: ComponentConfig = Field(
        default=..., description="The configuration for the training dataset."
    )

    val_dataset: ComponentConfig = Field(
        default=..., description="The configuration for the validation dataset."
    )

    test_dataset: ComponentConfig | None = Field(
        default=None, description="The configuration for the test dataset."
    )

    transforms: list[ComponentConfig] | None = Field(
        default_factory=list, description="List of transform component configurations."
    )

    global_dataloader: DataLoaderConfig = Field(
        default_factory=DataLoaderConfig, description="The global DataLoader configuration."
    )

    train_dataloader: DataLoaderConfig | None = Field(
        default=None, description="The DataLoader configuration for the training dataset."
    )
    val_dataloader: DataLoaderConfig | None = Field(
        default=None, description="The DataLoader configuration for the validation dataset."
    )
    test_dataloader: DataLoaderConfig | None = Field(
        default=None, description="The DataLoader configuration for the test dataset."
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
