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


@dataclass
class BackendCapabilities:
    """Backend traits declared at registration time.

    These declarations describe supported behavior and defaults. They are not
    host-resolved state and must never grant security exemptions.
    """

    execution_location: ExecutionLocation = ExecutionLocation.UNKNOWN
    filesystem_semantics: FilesystemSemantics = FilesystemSemantics.UNKNOWN
    accepts_host_cwd: bool = False
    requires_sandbox_cwd: bool = False
    supports_image: bool = False
    supports_resource_limits: bool = False
    supports_pty: bool = False
    supports_background_processes: bool = False
    supports_file_transfer: bool = False
    supports_persistence: bool = False


@dataclass
class EffectiveBackendCapabilities(BackendCapabilities):
    """Capabilities resolved by the host for one task request.

    The Environment Manager derives this state from the registered declaration,
    host configuration, task overrides, and the factory request. Plugin-declared
    values alone must not reduce approval requirements.
    """

    host_access: HostAccess = HostAccess.UNKNOWN


@dataclass
class EnvironmentRuntimeState:
    """Host-observed security-relevant state for an active environment."""

    backend_name: str
    task_id: str
    execution_location: ExecutionLocation = ExecutionLocation.UNKNOWN
    filesystem_semantics: FilesystemSemantics = FilesystemSemantics.UNKNOWN
    host_access: HostAccess = HostAccess.UNKNOWN
    isolation_verified_by_host: bool = False

    @property
    def has_verified_no_host_access(self) -> bool:
        """Return whether the host verified that this instance has no host access."""
        return self.isolation_verified_by_host and self.host_access is HostAccess.NONE


@dataclass
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
        self.terminal_config = dict(self.terminal_config)
        self.task_overrides = dict(self.task_overrides)
        self.backend_config = dict(self.backend_config)


BackendFactory = Callable[[BackendFactoryRequest], "BaseEnvironment"]
AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


_BACKEND_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass
class BackendDefinition:
    """Registration metadata for a terminal backend."""

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
    default_cwd: str = ""
    config_resolver: Callable[[], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _BACKEND_NAME_RE.fullmatch(self.name):
            raise ValueError("backend name must match ^[a-z][a-z0-9_-]*$")
        if not callable(self.factory):
            raise TypeError("backend factory must be callable")
        if not isinstance(self.default_cwd, str):
            raise TypeError("backend default_cwd must be a string")
        if not callable(self.availability_check):
            raise TypeError("availability_check must be callable")
        if self.config_resolver is not None and not callable(self.config_resolver):
            raise TypeError("config_resolver must be callable")
        if self.config_schema is not None:
            self.config_schema = dict(self.config_schema)
        self.diagnostic_metadata = dict(self.diagnostic_metadata)
        if not self.label:
            self.label = self.name

    def is_available(self) -> bool:
        """Return whether the backend can be constructed in this process."""
        return bool(self.availability_check())
