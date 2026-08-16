from aide.registry.registries import ModelRegistry


def test_model_registry_validator():
    try:

        @ModelRegistry.register("not_a_model")  # type: ignore
        class NotAModel:
            pass

    except TypeError as e:
        assert "must inherit from TrainableModel" in str(e)
