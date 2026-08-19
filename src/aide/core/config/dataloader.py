from __future__ import annotations

from pydantic import BaseModel, Field


class DataLoaderConfig(BaseModel):
    """Configuration for a data loader."""

    batch_size: int = Field(default=1, description="The batch size for the data loader.")

    shuffle: bool = Field(default=True, description="Whether to shuffle the data.")

    num_workers: int = Field(
        default=0, description="The number of worker threads for loading the data."
    )

    pin_memory: bool = Field(default=True, description="Whether to pin memory during data loading.")

    drop_last: bool = Field(default=False, description="Whether to drop the last incomplete batch.")

    persistent_workers: bool = Field(
        default=False, description="Whether to keep worker threads alive between epochs."
    )

    def override_with(self, other: DataLoaderConfig | None) -> "DataLoaderConfig":
        """
        Override the current configuration with another configuration.

        Args:
            other (DataLoaderConfig): The configuration to override with.

        Returns:
            DataLoaderConfig: The resulting configuration after overriding.
        """
        if other is None:
            return self
        return self.model_copy(update=other.model_dump())
