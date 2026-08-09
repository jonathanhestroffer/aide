from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ComponentConfig(BaseModel):
    """Configuration for a registry-backed component.

    Describes the configuration for a component that can be registered and instantiated,
    including the fully qualified class name and any parameters required for initialization.
    """

    class_name: str = Field(..., description="The fully qualified class name of the component.")

    params: dict[str, Any] = Field(
        default_factory=dict, description="The parameters for the component."
    )
