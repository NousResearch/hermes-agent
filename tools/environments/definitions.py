"""Public definitions for pluggable terminal backends.

This module is intentionally dependency-light so third-party plugins can import
backend contracts without importing the terminal tool implementation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
class BackendFactoryRequest:
    """Host-owned inputs passed to a backend factory.

    ``backend_config`` contains raw profile config before manager resolution and
    the resolver's defensive result by the time the backend factory receives it.
    """

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
ConfigAvailabilityCheck = Callable[[Mapping[str, Any]], bool]
ConfigResolver = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _always_available() -> bool:
    return True


_BACKEND_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_CONFIG_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_UNSAFE_CONFIG_PATH_SEGMENTS = frozenset({"__proto__", "constructor", "prototype"})
_DASHBOARD_CONFIG_FIELD_TYPES = frozenset(
    {"boolean", "list", "number", "secret", "select", "string", "text"}
)


def validate_dashboard_config_path(path: str, *, label: str) -> None:
    segments = path.split(".")
    if any(
        not _CONFIG_PATH_SEGMENT_RE.fullmatch(segment)
        or segment in _UNSAFE_CONFIG_PATH_SEGMENTS
        for segment in segments
    ):
        raise ValueError(f"{label} contains an invalid or unsafe path segment")


def _validated_config_schema(
    config_schema: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if config_schema is None:
        return None
    schema: dict[str, dict[str, Any]] = {}
    for key, field_schema in config_schema.items():
        if not isinstance(key, str) or not key:
            raise TypeError("backend config schema keys must be non-empty strings")
        validate_dashboard_config_path(key, label=f"backend config schema key {key!r}")
        if not isinstance(field_schema, Mapping):
            raise TypeError(f"backend config schema entry {key!r} must be a mapping")
        entry = dict(field_schema)
        if any(not isinstance(metadata_key, str) for metadata_key in entry):
            raise TypeError(
                f"backend config schema entry {key!r} metadata keys must be strings"
            )
        field_type = entry.get("type", "string")
        if (
            not isinstance(field_type, str)
            or field_type not in _DASHBOARD_CONFIG_FIELD_TYPES
        ):
            raise ValueError(
                f"backend config schema entry {key!r} has unsupported type "
                f"{field_type!r}"
            )
        options = entry.get("options")
        if field_type == "select" and (
            not isinstance(options, list)
            or any(not isinstance(option, str) for option in options)
        ):
            raise TypeError(
                f"backend config schema entry {key!r} options must be a list of strings"
            )
        config_key = entry.get("config_key")
        if config_key is not None and (
            not isinstance(config_key, str) or not config_key
        ):
            raise TypeError(
                f"backend config schema entry {key!r} config_key must be "
                "a non-empty string"
            )
        if config_key is not None:
            validate_dashboard_config_path(
                config_key,
                label=f"backend config schema entry {key!r} config_key",
            )
        try:
            json.dumps(entry, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"backend config schema entry {key!r} must be JSON-serializable"
            ) from exc
        schema[key] = entry
    return schema


@dataclass
class BackendDefinition:
    """Registration metadata for a terminal backend.

    ``config_schema`` maps backend-local field names to dashboard field
    descriptors. The web dashboard stores them below
    ``terminal.backends.<name>`` by default. A field may set ``config_key`` to
    alias another key within that same namespace; only a canonical built-in may
    use it to preserve an existing legacy config path. The dashboard strips the
    routing key before returning the schema.

    ``config_resolver`` receives a defensive snapshot of that backend-local raw
    profile mapping and returns the complete runtime config. The backend owns
    precedence among defaults, profile values, and environment overrides. Core
    uses the returned snapshot for both config-aware availability and factory
    construction; factories must not read profile config or environment again.
    """

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
    config_resolver: ConfigResolver | None = None
    config_availability_check: ConfigAvailabilityCheck | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _BACKEND_NAME_RE.fullmatch(self.name):
            raise ValueError("backend name must match ^[a-z][a-z0-9_-]*$")
        if not callable(self.factory):
            raise TypeError("backend factory must be callable")
        if not isinstance(self.default_cwd, str):
            raise TypeError("backend default_cwd must be a string")
        if not callable(self.availability_check):
            raise TypeError("availability_check must be callable")
        if self.config_availability_check is not None and not callable(
            self.config_availability_check
        ):
            raise TypeError("config_availability_check must be callable")
        if self.config_resolver is not None and not callable(self.config_resolver):
            raise TypeError("config_resolver must be callable")
        self.config_schema = _validated_config_schema(self.config_schema)
        self.diagnostic_metadata = dict(self.diagnostic_metadata)
        self.validated_picker_metadata()
        if not self.label:
            self.label = self.name

    def validated_picker_metadata(self) -> dict[str, str]:
        """Return a validated snapshot of mutable user-visible metadata."""
        metadata = {
            "label": self.label,
            "description": self.description,
            "install_hint": self.install_hint,
        }
        for field_name, value in metadata.items():
            if not isinstance(value, str):
                raise TypeError(f"backend {field_name} must be a string")
        return metadata

    def is_available(
        self, backend_config: Mapping[str, Any] | None = None
    ) -> bool:
        """Return whether the backend can be constructed in this process."""
        if backend_config is not None and self.config_availability_check is not None:
            return bool(self.config_availability_check(dict(backend_config)))
        return bool(self.availability_check())

    def validated_config_schema(self) -> dict[str, dict[str, Any]]:
        """Return a validated snapshot of mutable dashboard schema metadata."""
        return _validated_config_schema(self.config_schema) or {}
