"""Host-owned lifecycle manager for terminal backend environments.

The public signatures are defined in the contract phase. Runtime behavior will
be implemented behind EXP_BACKEND in later changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from tools.environments.definitions import (
    BackendCapabilities,
    BackendDefinition,
    BackendFactoryRequest,
)
from tools.environments.registry import (
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

    def resolve_backend(self, name: str) -> BackendDefinition:
        """Resolve the selected backend definition."""
        _not_implemented()

    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        """Create a new environment for a fully resolved request."""
        _not_implemented()

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

    def get_capabilities(self, task_id: str) -> BackendCapabilities:
        """Return host-consumed capabilities for a task's selected backend."""
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
