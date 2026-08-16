from aide.registry.registries import ComponentRegistry


def test_component_registry_validator():
    try:

        @ComponentRegistry.register("not_a_component")  # type: ignore
        class NotAComponent:
            pass

    except TypeError as e:
        assert "must inherit from Component" in str(e)
