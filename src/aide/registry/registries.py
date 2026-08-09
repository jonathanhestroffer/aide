from __future__ import annotations

from aide.core.components import Component, TransformComponent
from aide.core.registry import Registry
from aide.core.trainable import TrainableModel


def _validate_model_class(key: str, model_cls: type[TrainableModel]) -> None:
    if not issubclass(model_cls, TrainableModel):
        raise TypeError(
            f"Model '{key}' must inherit from TrainableModel. "
            f"Got: {model_cls.__module__}.{model_cls.__name__}"
        )


def _validate_component_class(key: str, component_cls: type[Component]) -> None:
    if not issubclass(component_cls, Component):
        raise TypeError(
            f"Component '{key}' must inherit from Component. "
            f"Got: {component_cls.__module__}.{component_cls.__name__}"
        )


def _validate_transform_class(key: str, transform_cls: type[TransformComponent]) -> None:
    if not issubclass(transform_cls, TransformComponent):
        raise TypeError(
            f"Transform '{key}' must inherit from TransformComponent. "
            f"Got: {transform_cls.__module__}.{transform_cls.__name__}"
        )


ModelRegistry = Registry[type[TrainableModel]](
    "models",
    validator=_validate_model_class,
)

ComponentRegistry = Registry[type[Component]](
    "components",
    validator=_validate_component_class,
)

TransformRegistry = Registry[type[TransformComponent]](
    "transforms",
    validator=_validate_transform_class,
)
