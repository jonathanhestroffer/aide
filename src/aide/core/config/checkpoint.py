from typing import Literal

from pydantic import BaseModel, Field


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing during training."""

    enabled: bool = Field(
        default=True, description="Whether to enable checkpointing during training."
    )

    monitor: str | None = Field(
        default="val_loss", description="The metric to monitor for checkpointing."
    )

    mode: Literal["min", "max"] = Field(
        default="min", description="The mode for the monitored metric ('min' or 'max')."
    )

    save_last: bool = Field(default=True, description="Whether to save the last checkpoint.")

    save_top_k: int = Field(default=1, description="The number of top checkpoints to save.")

    every_n_epochs: int | None = Field(
        default=None, description="Save a checkpoint every n epochs."
    )

    every_n_train_steps: int | None = Field(
        default=None, description="Save a checkpoint every n training steps."
    )

    dirpath: str | None = Field(default=None, description="The directory path to save checkpoints.")

    filename: str = Field(
        default="{epoch}-{step}-{val_loss:.4f}",
        description="The filename template for checkpoints.",
    )
