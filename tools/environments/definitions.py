"""Public definitions for pluggable terminal backends.

This module is intentionally dependency-light so third-party plugins can import
backend contracts without importing the terminal tool implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment


class ExecutionLocation(str, Enum):
    """Where a backend executes commands relative to the Hermes host."""

    UNKNOWN = "unknown"
    LOCAL = "local"
    REMOTE = "remote"


class FilesystemSemantics(str, Enum):
    """How the backend filesystem relates to the Hermes host filesystem."""

    UNKNOWN = "unknown"
    HOST = "host"
    SHARED = "shared"
    ISOLATED = "isolated"


class HostAccess(str, Enum):
    """Whether an environment may access resources on the Hermes host."""

    UNKNOWN = "unknown"
    NONE = "none"
    POSSIBLE = "possible"
    DIRECT = "direct"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Descriptive backend traits used for routing and user-facing metadata.

    Capabilities are descriptive only. They never grant security exemptions.
    """

    execution_location: ExecutionLocation = ExecutionLocation.UNKNOWN
    filesystem_semantics: FilesystemSemantics = FilesystemSemantics.UNKNOWN
    host_access: HostAccess = HostAccess.UNKNOWN
    accepts_host_cwd: bool = False
    requires_sandbox_cwd: bool = False
    supports_image: bool = False
    supports_resource_limits: bool = False
    supports_pty: bool = False
    supports_background_processes: bool = False
    supports_file_transfer: bool = False
    supports_persistence: bool = False


_IMMUTABLE_CONTRACT_VALUE_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    Path,
    Enum,
)


def _freeze_contract_value(value: Any, field_name: str) -> Any:
    """Recursively freeze JSON-like contract data.

    Rejecting unsupported object types keeps the plugin-facing contract
    genuinely immutable instead of merely making a shallow defensive copy.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} mapping keys must be strings")
            frozen[key] = _freeze_contract_value(nested_value, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_contract_value(item, field_name) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_contract_value(item, field_name) for item in value)
    if isinstance(value, _IMMUTABLE_CONTRACT_VALUE_TYPES):
        return value
    raise TypeError(
        f"{field_name} contains unsupported mutable value of type "
        f"{type(value).__name__}"
    )


def _freeze_contract_mapping(
    value: Mapping[str, Any], field_name: str
) -> Mapping[str, Any]:
    frozen = _freeze_contract_value(value, field_name)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class BackendFactoryRequest:
    """Host-owned inputs passed to a backend factory."""

    backend_name: str
    task_id: str = "default"
    cwd: str = ""
    timeout: int = 180
    image: str = ""
    host_cwd: str | None = None
    profile_name: str = ""
    hermes_home: Path | None = None
    terminal_config: Mapping[str, Any] = field(default_factory=dict)
    task_overrides: Mapping[str, Any] = field(default_factory=dict)
    backend_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_config",
            _freeze_contract_mapping(self.terminal_config, "terminal_config"),
        )
        object.__setattr__(
            self,
            "task_overrides",
            _freeze_contract_mapping(self.task_overrides, "task_overrides"),
        )
        object.__setattr__(
            self,
            "backend_config",
            _freeze_contract_mapping(self.backend_config, "backend_config"),
        )


BackendFactory = Callable[[BackendFactoryRequest], "BaseEnvironment"]
AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


_BACKEND_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class BackendDefinition:
    """Immutable registration metadata for a terminal backend."""

    name: str
    factory: BackendFactory
    label: str = ""
    description: str = ""
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    availability_check: AvailabilityCheck = _always_available
    config_schema: Mapping[str, Any] | None = None
    install_hint: str = ""
    diagnostic_metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    plugin_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _BACKEND_NAME_RE.fullmatch(self.name):
            raise ValueError("backend name must match ^[a-z][a-z0-9_-]*$")
        if not callable(self.factory):
            raise TypeError("backend factory must be callable")
        if not callable(self.availability_check):
            raise TypeError("availability_check must be callable")
        if self.config_schema is not None:
            object.__setattr__(
                self,
                "config_schema",
                _freeze_contract_mapping(self.config_schema, "config_schema"),
            )
        object.__setattr__(
            self,
            "diagnostic_metadata",
            _freeze_contract_mapping(
                self.diagnostic_metadata,
                "diagnostic_metadata",
            ),
        )
        if not self.label:
            object.__setattr__(self, "label", self.name)

    def is_available(self) -> bool:
        """Return whether the backend can be constructed in this process."""
        return bool(self.availability_check())
