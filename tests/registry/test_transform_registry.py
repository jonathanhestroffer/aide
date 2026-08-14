from aide.registry.registries import TransformRegistry


def test_transform_registry_validator():
    try:

        @TransformRegistry.register("not_a_transform")  # type: ignore
        class NotATransform:
            pass

    except TypeError as e:
        assert "must inherit from TransformComponent" in str(e)
