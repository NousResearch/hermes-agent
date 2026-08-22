"""Host-owned resolver and factory for terminal backend environments."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tools.environments.definitions import (
    BackendDefinition,
    BackendFactoryRequest,
    ExecutionLocation,
)
from tools.environments.registry import (
    BackendUnavailableError,
    TerminalBackendRegistry,
    terminal_backend_registry,
)

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment


class EnvironmentManager:
    """Resolve registered backends and create their environments."""

    def __init__(self, registry: TerminalBackendRegistry | None = None) -> None:
        self.registry = registry if registry is not None else terminal_backend_registry

        from tools.environments.builtin_backends import (
            register_builtin_terminal_backends,
        )

        register_builtin_terminal_backends(self.registry)

    def resolve_backend(
        self,
        name: str,
        *,
        backend_config: Mapping[str, Any] | None = None,
    ) -> BackendDefinition:
        """Resolve the selected backend definition."""
        definition = self.registry.require(name)
        resolved_config = self.resolve_backend_config(definition, backend_config or {})
        if not definition.is_available(resolved_config):
            raise BackendUnavailableError(f"Terminal backend {name!r} is unavailable")
        return definition

    @staticmethod
    def resolve_backend_config(
        definition: BackendDefinition,
        raw_backend_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve one defensive raw-config snapshot into runtime config."""
        raw_snapshot = deepcopy(dict(raw_backend_config))
        if definition.config_resolver is None:
            return raw_snapshot
        resolved_config = definition.config_resolver(deepcopy(raw_snapshot))
        if not isinstance(resolved_config, Mapping):
            raise TypeError(
                f"Terminal backend {definition.name!r} config_resolver must return a mapping"
            )
        return deepcopy(dict(resolved_config))

    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        """Create a new environment for a fully resolved request."""
        from tools.environments.base import BaseEnvironment
        from tools.environments.builtin_backends import (
            is_canonical_builtin_definition,
        )

        definition = self.registry.require(request.backend_name)
        resolved_config = self.resolve_backend_config(
            definition, request.backend_config
        )
        request.backend_config = resolved_config

        # Canonical built-ins construct directly and surface the backend-specific
        # constructor/configuration error. Their separate
        # check_terminal_requirements() path still performs the preflight.
        # Third-party definitions retain manager-enforced availability.
        if (
            not is_canonical_builtin_definition(definition)
            and not definition.is_available(resolved_config)
        ):
            raise BackendUnavailableError(
                f"Terminal backend {request.backend_name!r} is unavailable"
            )
        environment = definition.factory(request)
        if not isinstance(environment, BaseEnvironment):
            raise TypeError(
                f"Terminal backend {definition.name!r} factory must return BaseEnvironment"
            )
        if definition.capabilities.execution_location is not ExecutionLocation.LOCAL:
            from tools.environments.local import LocalEnvironment

            if isinstance(environment, LocalEnvironment):
                raise TypeError(
                    f"Terminal backend {definition.name!r} declares "
                    f"{definition.capabilities.execution_location.value} execution "
                    "but returned LocalEnvironment"
                )
        return environment
