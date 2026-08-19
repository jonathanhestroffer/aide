from aide.registry.registries import DatasetRegistry, ModelRegistry, TransformRegistry


def test_model_registry_validator():
    try:

        @ModelRegistry.register("not_a_trainable_model")  # type: ignore
        class NotATrainableModel:
            pass

    except TypeError as e:
        assert "must be or inherit from" in str(e)
        assert "TrainableModel" in str(e)


def test_transform_registry_validator():
    try:

        @TransformRegistry.register("not_a_transform")  # type: ignore
        class NotATransform:
            pass

    except TypeError as e:
        assert "must be or inherit from" in str(e)
        assert "TransformComponent" in str(e)


def test_dataset_registry_validator():
    try:

        @DatasetRegistry.register("not_a_dataset")  # type: ignore
        class NotADataset:
            pass

    except TypeError as e:
        assert "must be or inherit from" in str(e)
        assert "Dataset" in str(e)
