from aide.core.components import Component, TransformComponent
from aide.core.config.component import ComponentConfig
from aide.registry.registries import ComponentRegistry, TransformRegistry


def build_component(config: ComponentConfig) -> Component:
    """Create a component from typed experiment config."""
    cls = ComponentRegistry.get(config.class_name)
    return cls(**config.params)


def build_transform_component(config: ComponentConfig) -> TransformComponent:
    """Create a transform component from typed experiment config."""
    cls = TransformRegistry.get(config.class_name)
    return cls(**config.params)
