from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic key ->  object registry with decorator-based registration."""

    def __init__(
        self,
        name: str,
        *,
        allow_override: bool = False,
        validator: Callable[[str, T], None] | None = None,
    ) -> None:
        """
        Args:
            name: Name of the registry, used in error messages.
            allow_override: If True, allow overriding existing keys.
            expected_type: Optional runtime type check for registered objects.
            validator: Optional registry-specific validation hook.
        """
        self.name = name
        self.allow_override = allow_override
        self.validator = validator
        self._objects: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Register an object under ``key`` and return it unchanged.

        This allows usage as:

        @registry.register("my_key")
        class MyObject:
                ...
        """

        def decorator(obj: T) -> T:
            self.add(key, obj)
            return obj

        return decorator

    def add(self, key: str, obj: T) -> None:
        """Register ``obj`` under ``key``."""
        if not key:
            raise ValueError(f"Registry '{self.name}' keys must be non-empty")

        if self.validator is not None:
            self.validator(key, obj)

        if key in self._objects and not self.allow_override:
            raise KeyError(
                f"Key '{key}' already exists in registry '{self.name}'. "
                "Set allow_override=True to replace existing entries."
            )

        self._objects[key] = obj

    def get(self, key: str) -> T:
        """Return registered object for ``key`` or raise helpful KeyError."""
        try:
            return self._objects[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._objects)) or "<empty>"
            raise KeyError(
                f"Unknown key '{key}' in registry '{self.name}'. Available keys: {available}"
            ) from exc

    def try_get(self, key: str) -> T | None:
        """Return registered object for ``key`` or ``None`` if absent."""
        return self._objects.get(key)

    def unregister(self, key: str) -> T:
        """Remove and return object associated with ``key``."""
        if key not in self._objects:
            raise KeyError(f"Cannot remove unknown key '{key}' from registry '{self.name}'")
        return self._objects.pop(key)

    def clear(self) -> None:
        """Remove all registered entries."""
        self._objects.clear()

    def keys(self) -> list[str]:
        """Return registered keys in insertion order."""
        return list(self._objects.keys())

    def values(self) -> list[T]:
        """Return registered values in insertion order."""
        return list(self._objects.values())

    def items(self) -> list[tuple[str, T]]:
        """Return (key, value) pairs in insertion order."""
        return list(self._objects.items())

    def __contains__(self, key: str) -> bool:
        return key in self._objects

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        keys = ", ".join(self.keys())
        return f"Registry(name={self.name!r}, keys=[{keys}])"
