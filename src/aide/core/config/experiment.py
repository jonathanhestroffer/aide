from __future__ import annotations

from pydantic import BaseModel, Field

from aide.core.config.checkpoint import CheckpointConfig
from aide.core.config.datamodule import DataModuleConfig
from aide.core.config.infrastructure import InfrastructureConfig
from aide.core.config.trainable import TrainableConfig
from aide.core.config.trainer import TrainerConfig


class ExperimentMetadata(BaseModel):
    """Metadata for an ML experiment.

    Describes the metadata for an ML experiment, including the
    name, description, and tags.
    """

    name: str = Field(..., description="The name of the experiment.")
    description: str | None = Field(None, description="A brief description of the experiment.")
    tags: list[str] = Field(default_factory=list, description="A list of tags for the experiment.")


class ExperimentConfig(BaseModel):
    """Configuration for an ML experiment.

    Describes the configuration for an ML experiment, including the
    trainable pipeline, data module, trainer, and infrastructure settings.
    """

    metadata: ExperimentMetadata = Field(..., description="The metadata for the experiment.")

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
