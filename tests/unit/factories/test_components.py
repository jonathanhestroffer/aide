from aide.components.factory import build_component, build_transform_component
from aide.core.components import Component, TransformComponent
from aide.core.config.component import ComponentConfig
from aide.registry.registries import ComponentRegistry, TransformRegistry


def test_build_component():

    # Register a dummy model class for testing
    class ExampleComponent(Component):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

        def forward(self, *args, **kwargs):
            return

    ComponentRegistry.add("ExampleComponent", ExampleComponent)

    component_config = ComponentConfig(
        class_name="ExampleComponent", params={"param1": 10, "param2": 20}
    )

    component_instance = build_component(component_config)

    # Assertions to verify the component instance is created correctly
    assert isinstance(component_instance, ExampleComponent)
    assert component_instance.param1 == 10
    assert component_instance.param2 == 20


def test_build_transform_component():

    # Register a dummy model class for testing
    class ExampleTransformComponent(TransformComponent):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

        def forward(self, *args, **kwargs):
            return

    TransformRegistry.add("ExampleTransformComponent", ExampleTransformComponent)

    component_config = ComponentConfig(
        class_name="ExampleTransformComponent", params={"param1": 10, "param2": 20}
    )

    component_instance = build_transform_component(component_config)

    # Assertions to verify the component instance is created correctly
    assert isinstance(component_instance, ExampleTransformComponent)
    assert component_instance.param1 == 10
    assert component_instance.param2 == 20
