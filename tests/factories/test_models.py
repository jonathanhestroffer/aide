from aide.core.config.component import ComponentConfig
from aide.core.config.trainable import TrainableConfig
from aide.core.trainable import TrainableModel
from aide.models.factory import build_trainable_model
from aide.registry.registries import ModelRegistry


def test_build_trainable_model():

    # Register a dummy model class for testing
    class ExampleTrainableModel(TrainableModel):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

        def forward(self, *args, **kwargs):
            return

        def training_step(self, *args, **kwargs):
            return

        def configure_optimizers(self, *args, **kwargs):
            return

    ModelRegistry.add("ExampleTrainableModel", ExampleTrainableModel)

    # Build the trainable model using the factory function
    trainable_config = TrainableConfig(
        model=ComponentConfig(
            class_name="ExampleTrainableModel", params={"param1": 10, "param2": 20}
        ),
        preprocessor=None,
        postprocessor=None,
    )

    model_instance = build_trainable_model(trainable_config)

    # Assertions to verify the model instance is created correctly
    assert isinstance(model_instance, ExampleTrainableModel)
    assert model_instance.param1 == 10
    assert model_instance.param2 == 20
