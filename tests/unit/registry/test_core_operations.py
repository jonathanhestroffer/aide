from aide.core.registry import Registry


def test_registry_basic_operations():
    registry = Registry("test_registry")

    # Test adding and retrieving an object
    registry.add("key1", "value1")
    assert registry.get("key1") == "value1"

    # Test try_get method
    assert registry.try_get("key1") == "value1"
    assert registry.try_get("nonexistent_key") is None

    # Test unregistering an object
    removed_value = registry.unregister("key1")
    assert removed_value == "value1"
    assert registry.try_get("key1") is None

    # Test clearing the registry
    registry.add("key2", "value2")
    registry.clear()
    assert len(registry.keys()) == 0


def test_registry_key_validation():
    def validator(key, obj):
        if not isinstance(obj, str):
            raise ValueError("Object must be a string")

    registry = Registry("test_registry", validator=validator)

    # Test adding a valid object
    registry.add("key1", "value1")
    assert registry.get("key1") == "value1"

    # Test adding an invalid object
    try:
        registry.add("key2", 123)  # Not a string
    except ValueError as e:
        assert str(e) == "Object must be a string"


def test_registry_allow_override():
    registry = Registry("test_registry", allow_override=True)

    # Test adding an object and overriding it
    registry.add("key1", "value1")
    assert registry.get("key1") == "value1"

    registry.add("key1", "new_value")
    assert registry.get("key1") == "new_value"


def test_registry_no_override():
    registry = Registry("test_registry", allow_override=False)

    # Test adding an object and attempting to override it
    registry.add("key1", "value1")
    assert registry.get("key1") == "value1"

    try:
        registry.add("key1", "new_value")
    except KeyError as e:
        assert "already exists in registry" in str(e)


def test_registry_decorator_with_function():
    registry = Registry("test_registry")

    @registry.register("key1")
    def my_function():
        return "Hello, World!"

    assert registry.get("key1")() == "Hello, World!"


def test_registry_empty_key():
    registry = Registry("test_registry")

    try:
        registry.add("", "value")
    except ValueError as e:
        assert "keys must be non-empty" in str(e)


def test_registry_unknown_key():
    registry = Registry("test_registry")

    try:
        registry.get("unknown_key")
    except KeyError as e:
        assert "Unknown key" in str(e)


def test_registry_keys_and_values():
    registry = Registry("test_registry")

    registry.add("key1", "value1")
    registry.add("key2", "value2")

    assert set(registry.keys()) == {"key1", "key2"}
    assert set(registry.values()) == {"value1", "value2"}


def test_registry_unregister_unknown_key():
    registry = Registry("test_registry")

    try:
        registry.unregister("unknown_key")
    except KeyError as e:
        assert "Cannot remove unknown key" in str(e)


def test_registry_clear():
    registry = Registry("test_registry")

    registry.add("key1", "value1")
    registry.add("key2", "value2")

    registry.clear()
    assert len(registry.keys()) == 0
    assert len(registry.values()) == 0


def test_registry_decorator_with_class():
    registry = Registry("test_registry")

    @registry.register("my_key")
    class MyObject:
        def greet(self):
            return "Hello from MyObject!"

    obj = registry.get("my_key")()
    assert obj.greet() == "Hello from MyObject!"
