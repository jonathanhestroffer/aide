from torch.utils.data import Dataset

from aide.components.factory import build_dataset, build_trainable, build_transform
from aide.core.components import TransformComponent
from aide.core.config.component import ComponentConfig
from aide.core.trainable import TrainableModel
from aide.registry.registries import DatasetRegistry, ModelRegistry, TransformRegistry


def test_build_transform():

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

    component_instance = build_transform(component_config)

    # Assertions to verify the component instance is created correctly
    assert isinstance(component_instance, ExampleTransformComponent)
    assert component_instance.param1 == 10
    assert component_instance.param2 == 20


def test_build_model():

    # Register a dummy model class for testing
    class ExampleModelComponent(TrainableModel):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def forward(self, *args, **kwargs):
            return

        def training_step(self, *args, **kwargs):
            return

        def configure_optimizers(self, *args, **kwargs):
            return

    ModelRegistry.add("ExampleModelComponent", ExampleModelComponent)

    component_config = ComponentConfig(
        class_name="ExampleModelComponent", params={"param1": 10, "param2": 20}
    )

    component_instance = build_trainable(component_config)

    # Assertions to verify the component instance is created correctly
    assert isinstance(component_instance, ExampleModelComponent)
    assert component_instance.param1 == 10
    assert component_instance.param2 == 20


def test_build_dataset():

    # Register a dummy dataset class for testing
    class ExampleDataset(Dataset):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def __len__(self):
            return 0

        def __getitem__(self, index):
            return None

    DatasetRegistry.add("ExampleDataset", ExampleDataset)

    component_config = ComponentConfig(
        class_name="ExampleDataset", params={"param1": 10, "param2": 20}
    )

    component_instance = build_dataset(component_config)

    # Assertions to verify the component instance is created correctly
    assert isinstance(component_instance, ExampleDataset)
    assert component_instance.param1 == 10
    assert component_instance.param2 == 20
