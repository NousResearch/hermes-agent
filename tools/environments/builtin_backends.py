"""Host-owned registration for built-in terminal backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.environments.definitions import (
    BackendCapabilities,
    BackendDefinition,
    BackendFactoryRequest,
    ExecutionLocation,
    FilesystemSemantics,
)

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment
    from tools.environments.registry import TerminalBackendRegistry


# Reserve every legacy built-in name before its definition is migrated. This
# keeps the name-based legacy behavior stable without allowing third-party
# plugins to claim a built-in identity during the incremental cutover.
RESERVED_BUILTIN_BACKEND_NAMES = frozenset(
    {
        "local",
        "docker",
        "singularity",
        "modal",
        "managed_modal",
        "daytona",
        "ssh",
        "vercel_sandbox",
    }
)


def _create_local_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    from tools.environments.local import LocalEnvironment

    return LocalEnvironment(cwd=request.cwd, timeout=request.timeout)


def _local_backend_definition() -> BackendDefinition:
    return BackendDefinition(
        name="local",
        label="Local",
        description="Execute commands directly on the Hermes host.",
        factory=_create_local_environment,
        capabilities=BackendCapabilities(
            execution_location=ExecutionLocation.LOCAL,
            filesystem_semantics=FilesystemSemantics.HOST,
            accepts_host_cwd=True,
            requires_sandbox_cwd=False,
            supports_image=False,
            supports_resource_limits=False,
            supports_pty=True,
            supports_background_processes=True,
            supports_file_transfer=False,
            supports_persistence=True,
        ),
        source="builtin",
    )


def register_builtin_terminal_backends(registry: "TerminalBackendRegistry") -> None:
    """Register built-ins, rejecting any altered definition of a reserved name."""
    canonical = _local_backend_definition()
    existing = registry.get(canonical.name)
    if existing is None:
        registry.register(canonical)
        return

    if existing != canonical:
        from tools.environments.registry import BackendAlreadyRegisteredError

        raise BackendAlreadyRegisteredError(
            "Terminal backend 'local' conflicts with the built-in local backend"
        )
