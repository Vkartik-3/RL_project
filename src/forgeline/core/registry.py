"""A tiny named registry used for algorithms, rewards, verifiers, evaluators, etc."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, TypeVar

from forgeline.core.errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string keys to factories/classes with helpful error messages."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T | None = None) -> Callable[[T], T] | T:
        if item is not None:
            self._add(name, item)
            return item

        def decorator(obj: T) -> T:
            self._add(name, obj)
            return obj

        return decorator

    def _add(self, name: str, item: T) -> None:
        if name in self._items:
            raise ConfigError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(
                f"Unknown {self.kind} {name!r}. Registered: {sorted(self._items)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def names(self) -> list[str]:
        return sorted(self._items)
