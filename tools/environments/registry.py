"""Registry for terminal backend definitions.

The registry owns definitions only. Live environment instances belong to the
host-owned EnvironmentManager.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Callable

from hermes_constants import get_hermes_home

from tools.environments.definitions import BackendDefinition


class BackendAlreadyRegisteredError(ValueError):
    """Raised when a different backend definition claims an existing name."""


class BackendNotFoundError(LookupError):
    """Raised when a requested terminal backend is not registered."""


class BackendUnavailableError(RuntimeError):
    """Raised when a registered terminal backend is unavailable."""


class BackendDefinitionMutatedError(RuntimeError):
    """Raised when a registered definition no longer matches its registry key."""


class TerminalBackendRegistry:
    """Thread-safe collection of terminal backend definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, BackendDefinition] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_key(name: str, definition: BackendDefinition) -> None:
        if definition.name != name:
            raise BackendDefinitionMutatedError(
                f"Terminal backend registered as {name!r} was mutated to "
                f"{definition.name!r}"
            )
        definition.validated_picker_metadata()

    def _validate_all_locked(self) -> None:
        for name, definition in self._definitions.items():
            self._validate_key(name, definition)

    def register(self, definition: BackendDefinition) -> BackendDefinition:
        """Register a definition, rejecting invalid or duplicate entries."""
        if not isinstance(definition, BackendDefinition):
            raise TypeError("registry entries must be BackendDefinition instances")
        with self._lock:
            self._validate_all_locked()
            self._validate_key(definition.name, definition)
            if definition.name in self._definitions:
                raise BackendAlreadyRegisteredError(
                    f"Terminal backend {definition.name!r} is already registered"
                )
            self._definitions[definition.name] = definition
            return definition

    def register_or_verify(self, definition: BackendDefinition) -> BackendDefinition:
        """Atomically register a definition or return an exact existing match."""
        if not isinstance(definition, BackendDefinition):
            raise TypeError("registry entries must be BackendDefinition instances")
        with self._lock:
            self._validate_all_locked()
            self._validate_key(definition.name, definition)
            existing = self._definitions.get(definition.name)
            if existing is None:
                self._definitions[definition.name] = definition
                return definition
            if existing != definition:
                raise BackendAlreadyRegisteredError(
                    f"Terminal backend {definition.name!r} conflicts with the "
                    "registered definition"
                )
            return existing

    def get(self, name: str) -> BackendDefinition | None:
        """Return a definition by name, or None when it is absent."""
        with self._lock:
            self._validate_all_locked()
            return self._definitions.get(name)

    def require(self, name: str) -> BackendDefinition:
        """Return a definition by name or raise BackendNotFoundError."""
        definition = self.get(name)
        if definition is None:
            raise BackendNotFoundError(f"Terminal backend {name!r} is not registered")
        return definition

    def list_definitions(self) -> tuple[BackendDefinition, ...]:
        """Return registered definitions in deterministic name order."""
        with self._lock:
            self._validate_all_locked()
            return tuple(self._definitions[name] for name in sorted(self._definitions))

    def reset(self) -> None:
        """Remove all definitions for test isolation and controlled reloads."""
        with self._lock:
            self._definitions.clear()

    def unregister(self, name: str) -> BackendDefinition | None:
        """Remove and return one definition, if it is registered."""
        with self._lock:
            return self._definitions.pop(name, None)

    def unregister_if_same(
        self, name: str, registered: BackendDefinition
    ) -> BackendDefinition | None:
        """Remove a registration only if its private snapshot still owns the key."""
        with self._lock:
            if self._definitions.get(name) is not registered:
                return None
            return self._definitions.pop(name)


class ProfileScopedTerminalBackendRegistry:
    """Resolve one backend registry per active Hermes profile/home."""

    def __init__(self, key_resolver: Callable[[], Path] = get_hermes_home) -> None:
        self._key_resolver = key_resolver
        self._registries: dict[Path, TerminalBackendRegistry] = {}
        self._lock = RLock()

    def current(self) -> TerminalBackendRegistry:
        key = self._key_resolver().expanduser().resolve()
        with self._lock:
            registry = self._registries.get(key)
            if registry is None:
                registry = TerminalBackendRegistry()
                self._registries[key] = registry
            return registry

    def register(self, definition: BackendDefinition) -> BackendDefinition:
        return self.current().register(definition)

    def register_or_verify(self, definition: BackendDefinition) -> BackendDefinition:
        return self.current().register_or_verify(definition)

    def get(self, name: str) -> BackendDefinition | None:
        return self.current().get(name)

    def require(self, name: str) -> BackendDefinition:
        return self.current().require(name)

    def list_definitions(self) -> tuple[BackendDefinition, ...]:
        return self.current().list_definitions()

    def reset(self) -> None:
        self.current().reset()

    def unregister(self, name: str) -> BackendDefinition | None:
        return self.current().unregister(name)


terminal_backend_registry = ProfileScopedTerminalBackendRegistry()


def current_terminal_backend_registry() -> TerminalBackendRegistry:
    """Return the concrete registry for the active profile."""
    registry = terminal_backend_registry
    if isinstance(registry, ProfileScopedTerminalBackendRegistry):
        return registry.current()
    return registry
