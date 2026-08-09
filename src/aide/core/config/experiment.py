from __future__ import annotations

from pydantic import BaseModel, Field

from aide.core.config.checkpoint import CheckpointConfig
from aide.core.config.datamodule import DataModuleConfig
from aide.core.config.infrastructure import InfrastructureConfig
from aide.core.config.trainable import TrainableConfig
from aide.core.config.trainer import TrainerConfig


class ExperimentConfig(BaseModel):
    """Configuration for an ML experiment.

    Describes the configuration for an ML experiment, including the
    trainable pipeline, data module, trainer, and infrastructure settings.
    """

    trainable: TrainableConfig = Field(..., description="The trainable pipeline configuration.")

    datamodule: DataModuleConfig = Field(..., description="The data module configuration.")

    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig, description="The trainer configuration."
    )

    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="The checkpoint configuration."
    )

    infrastructure: InfrastructureConfig = Field(
        ..., description="The infrastructure configuration."
    )
