"""Registry for terminal backend definitions.

The registry owns definitions only. Live environment instances belong to the
host-owned EnvironmentManager.
"""

from __future__ import annotations

from threading import RLock

from tools.environments.definitions import BackendDefinition


class BackendAlreadyRegisteredError(ValueError):
    """Raised when a different backend definition claims an existing name."""


class BackendNotFoundError(LookupError):
    """Raised when a requested terminal backend is not registered."""


class TerminalBackendRegistry:
    """Thread-safe collection of immutable terminal backend definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, BackendDefinition] = {}
        self._lock = RLock()

    def register(self, definition: BackendDefinition) -> None:
        """Register a definition, rejecting invalid or duplicate entries."""
        if not isinstance(definition, BackendDefinition):
            raise TypeError("registry entries must be BackendDefinition instances")
        with self._lock:
            if definition.name in self._definitions:
                raise BackendAlreadyRegisteredError(
                    f"Terminal backend {definition.name!r} is already registered"
                )
            self._definitions[definition.name] = definition

    def get(self, name: str) -> BackendDefinition | None:
        """Return a definition by name, or None when it is absent."""
        with self._lock:
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
            return tuple(self._definitions[name] for name in sorted(self._definitions))

    def reset(self) -> None:
        """Remove all definitions for test isolation and controlled reloads."""
        with self._lock:
            self._definitions.clear()

    def unregister(self, name: str) -> BackendDefinition | None:
        """Remove and return one definition, if it is registered."""
        with self._lock:
            return self._definitions.pop(name, None)


terminal_backend_registry = TerminalBackendRegistry()
