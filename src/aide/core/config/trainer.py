from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """Configuration for the PyTorch Lightning Trainer.

    Defines the configuration for the PyTorch Lightning Trainer, including parameters such as
    maximum epochs, accelerator type, number of devices, and precision settings.
    """

    max_epochs: int = Field(default=10, description="Maximum number of epochs to train for.")

    accelerator: str = Field(
        default="auto", description="Device accelerator to use (e.g., 'cpu', 'gpu', 'auto')."
    )

    devices: int | str = Field(
        default="auto", description="Number of devices to use (e.g., 1, 2, 'auto')."
    )

    precision: str | None = Field(
        default=None, description="Precision to use for training (e.g., '16', '32', 'bf16')."
    )

    deterministic: bool | Literal["warn"] | None = Field(
        default=False,
        description="Whether to enable deterministic training for reproducibility.",
    )
