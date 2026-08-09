from __future__ import annotations

from pydantic import BaseModel, Field

from aide.core.config.component import ComponentConfig


class TrainableConfig(BaseModel):
    """Configuration for the trainable pipeline around a model component."""

    model: ComponentConfig = Field(..., description="The model component configuration.")

    preprocessor: ComponentConfig | None = Field(
        None, description="The preprocessor component configuration."
    )

    postprocessor: ComponentConfig | None = Field(
        None, description="The postprocessor component configuration."
    )
