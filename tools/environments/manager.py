"""Host-owned lifecycle manager for terminal backend environments.

The first EXP_BACKEND migration slice implements registry resolution and
factory-backed creation. Task-scoped reuse, overrides, runtime state, and
cleanup remain explicit stubs until their callers migrate together.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from tools.environments.definitions import (
    BackendDefinition,
    BackendFactoryRequest,
    EffectiveBackendCapabilities,
    EnvironmentRuntimeState,
    ExecutionLocation,
)
from tools.environments.registry import (
    BackendUnavailableError,
    TerminalBackendRegistry,
    terminal_backend_registry,
)

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment


_UNIMPLEMENTED = "experimental backend runtime is not implemented"


def _not_implemented() -> NoReturn:
    raise NotImplementedError(_UNIMPLEMENTED)


class EnvironmentManager:
    """Own environment resolution, creation, reuse, overrides, and cleanup."""

    def __init__(self, registry: TerminalBackendRegistry | None = None) -> None:
        self.registry = registry if registry is not None else terminal_backend_registry

        from tools.environments.builtin_backends import (
            register_builtin_terminal_backends,
        )

        register_builtin_terminal_backends(self.registry)

    def resolve_backend(self, name: str) -> BackendDefinition:
        """Resolve the selected backend definition."""
        definition = self.registry.require(name)
        if not definition.is_available():
            raise BackendUnavailableError(f"Terminal backend {name!r} is unavailable")
        return definition

    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        """Create a new environment for a fully resolved request."""
        from tools.environments.base import BaseEnvironment

        definition = self.resolve_backend(request.backend_name)
        resolved_config: dict[str, Any] = {}
        if definition.config_resolver is not None:
            plugin_config = definition.config_resolver()
            if not isinstance(plugin_config, Mapping):
                raise TypeError(
                    f"Terminal backend {definition.name!r} config_resolver must return a mapping"
                )
            resolved_config.update(plugin_config)
        resolved_config.update(request.backend_config)
        request.backend_config = resolved_config

        environment = definition.factory(request)
        if not isinstance(environment, BaseEnvironment):
            raise TypeError(
                f"Terminal backend {definition.name!r} factory must return BaseEnvironment"
            )
        if definition.capabilities.execution_location is ExecutionLocation.REMOTE:
            from tools.environments.local import LocalEnvironment

            if isinstance(environment, LocalEnvironment):
                raise TypeError(
                    f"Terminal backend {definition.name!r} declares remote execution "
                    "but returned LocalEnvironment"
                )
        return environment

    def get_or_create_environment(
        self, request: BackendFactoryRequest
    ) -> "BaseEnvironment":
        """Return the task-scoped environment, creating it when absent."""
        _not_implemented()

    def get_active_environment(self, task_id: str) -> "BaseEnvironment | None":
        """Return the active environment for a task, if one exists."""
        _not_implemented()

    def get_effective_backend_name(self, task_id: str) -> str:
        """Return the backend selected for a task."""
        _not_implemented()

    def get_effective_capabilities(self, task_id: str) -> EffectiveBackendCapabilities:
        """Return host-resolved capabilities for a task's selected backend."""
        _not_implemented()

    def get_runtime_state(self, task_id: str) -> EnvironmentRuntimeState | None:
        """Return host-observed state for an active task environment."""
        _not_implemented()

    def register_task_overrides(
        self, task_id: str, overrides: Mapping[str, Any]
    ) -> None:
        """Register task-scoped backend configuration overrides."""
        _not_implemented()

    def resolve_task_overrides(self, task_id: str) -> Mapping[str, Any]:
        """Resolve effective task-scoped backend configuration overrides."""
        _not_implemented()

    def clear_task_overrides(self, task_id: str) -> None:
        """Remove task-scoped backend configuration overrides."""
        _not_implemented()

    def mark_activity(self, task_id: str) -> None:
        """Record activity for idle-lifecycle decisions."""
        _not_implemented()

    def cleanup_environment(self, task_id: str) -> None:
        """Clean up and forget the environment owned by one task."""
        _not_implemented()

    def cleanup_all(self) -> None:
        """Clean up every environment owned by this manager."""
        _not_implemented()

    def snapshot(self) -> Mapping[str, Any]:
        """Return a read-only diagnostic snapshot of manager state."""
        _not_implemented()
