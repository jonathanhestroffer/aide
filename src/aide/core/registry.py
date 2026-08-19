from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")  # Method-level TypeVar to preserve decorated object types


class Registry(Generic[T]):
    """Generic key -> object registry with decorator-based registration."""

    def __init__(
        self,
        name: str,
        *,
        allow_override: bool = False,
        expected_type: type | None = None,
    ) -> None:
        self.name = name
        self.allow_override = allow_override
        self.expected_type = expected_type
        self._objects: dict[str, T] = {}

    def register(self, key: str) -> Callable[[R], R]:
        """Register an object under ``key`` and return it unchanged."""

        def decorator(obj: R) -> R:
            self.add(key, obj)  # type: ignore[arg-type]
            return obj

        return decorator

    def add(self, key: str, obj: T) -> None:
        """Register ``obj`` under ``key``."""
        if not key:
            raise ValueError(f"Registry '{self.name}' keys must be non-empty")

        if self.expected_type is not None:
            # Handle class registration (issubclass) vs instance registration (isinstance)
            is_valid = (
                issubclass(obj, self.expected_type)
                if isinstance(obj, type) and isinstance(self.expected_type, type)
                else isinstance(obj, self.expected_type)
            )
            if not is_valid:
                raise TypeError(
                    f"Object registered under key '{key}' in registry '{self.name}' "
                    f"must be or inherit from "
                    f"{self.expected_type.__module__}.{self.expected_type.__name__}. "
                    f"Got: {type(obj).__module__}.{type(obj).__name__}"
                )

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
