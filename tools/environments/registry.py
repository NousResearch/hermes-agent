"""Public registration contract for third-party terminal backends.

Product-specific backends belong in standalone plugins. Hermes owns backend
selection, lifecycle, and tool routing; plugins only provide an immutable
definition and a factory that returns a :class:`BaseEnvironment` compatible
object.
"""

from __future__ import annotations

import re
import threading
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping


BUILTIN_TERMINAL_BACKENDS = frozenset(
    {
        "local",
        "docker",
        "singularity",
        "modal",
        "daytona",
        "vercel_sandbox",
        "ssh",
    }
)

_BACKEND_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _freeze_config_value(value: Any) -> Any:
    """Return an independent, recursively read-only config snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _freeze_config_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_config_value(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class TerminalBackendRequest:
    """Host-owned input passed to a terminal backend factory.

    The mappings are read-only snapshots. A plugin cannot change Hermes
    configuration or another task's overrides through this object.
    """

    name: str
    task_id: str
    cwd: str
    timeout: int
    image: str
    settings: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        name: str,
        task_id: str,
        cwd: str,
        timeout: int,
        image: str,
        settings: Mapping[str, Any] | None = None,
    ) -> "TerminalBackendRequest":
        return cls(
            name=name,
            task_id=task_id,
            cwd=cwd,
            timeout=timeout,
            image=image,
            settings=_freeze_config_value(dict(settings or {})),
        )


TerminalBackendFactory = Callable[[TerminalBackendRequest], Any]


@dataclass(frozen=True, slots=True)
class TerminalBackendDefinition:
    """Immutable description of one plugin-provided terminal backend."""

    name: str
    factory: TerminalBackendFactory
    container_paths: bool = True
    default_cwd: str = "/root"
    default_image: str = ""
    image_override_key: str | None = None
    image_config_key: str | None = "image"

    def __post_init__(self) -> None:
        if not _BACKEND_NAME_RE.fullmatch(self.name):
            raise ValueError(
                "Terminal backend names must start with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores"
            )
        if self.name in BUILTIN_TERMINAL_BACKENDS:
            raise ValueError(f"Terminal backend name {self.name!r} is reserved by Hermes")
        if not callable(self.factory):
            raise TypeError("Terminal backend factory must be callable")
        if not self.default_cwd or not (
            self.default_cwd == "~" or self.default_cwd.startswith("/")
        ):
            raise ValueError("Terminal backend default_cwd must be '~' or an absolute path")
        if self.image_override_key and not _BACKEND_NAME_RE.fullmatch(
            self.image_override_key
        ):
            raise ValueError("image_override_key must be a lowercase configuration key")
        if self.image_config_key and not _BACKEND_NAME_RE.fullmatch(
            self.image_config_key
        ):
            raise ValueError("image_config_key must be a lowercase configuration key")
    def resolve_image(
        self,
        overrides: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> str:
        """Resolve a task override, plugin setting, then the default."""
        if self.image_override_key:
            override = overrides.get(self.image_override_key)
            if isinstance(override, str) and override.strip():
                return override.strip()
        if self.image_config_key:
            configured = settings.get(self.image_config_key)
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
        return self.default_image


class TerminalBackendRegistry:
    """Thread-safe registry for enabled plugin terminal backends."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._definitions: dict[str, tuple[TerminalBackendDefinition, str]] = {}

    def register(self, definition: TerminalBackendDefinition, *, owner: str) -> None:
        if not isinstance(definition, TerminalBackendDefinition):
            raise TypeError("register_terminal_backend expects TerminalBackendDefinition")
        owner = str(owner or "").strip()
        if not owner:
            raise ValueError("Terminal backend registration requires a plugin owner")

        # Keep a host-owned frozen copy. This prevents a plugin from retaining a
        # mutable definition object that changes registry behavior after load.
        hosted = replace(definition)
        with self._lock:
            existing = self._definitions.get(hosted.name)
            if existing is not None and existing[1] != owner:
                raise ValueError(
                    f"Terminal backend {hosted.name!r} is already registered by "
                    f"plugin {existing[1]!r}"
                )
            self._definitions[hosted.name] = (hosted, owner)

    def get(self, name: str) -> TerminalBackendDefinition | None:
        with self._lock:
            entry = self._definitions.get(str(name).lower())
            return entry[0] if entry else None

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._definitions))

    def owner(self, name: str) -> str | None:
        """Return the plugin id that registered *name*."""
        with self._lock:
            entry = self._definitions.get(str(name).lower())
            return entry[1] if entry else None

    def names_for_owner(self, owner: str) -> tuple[str, ...]:
        """Return the backend names currently owned by one plugin."""
        owner = str(owner or "").strip()
        with self._lock:
            return tuple(
                sorted(
                    name
                    for name, (_definition, registered_owner) in self._definitions.items()
                    if registered_owner == owner
                )
            )

    def snapshot(
        self,
    ) -> tuple[tuple[str, TerminalBackendDefinition, str], ...]:
        """Capture host-owned state for transactional plugin discovery."""
        with self._lock:
            return tuple(
                (name, definition, owner)
                for name, (definition, owner) in self._definitions.items()
            )

    def restore(
        self,
        snapshot: tuple[tuple[str, TerminalBackendDefinition, str], ...],
    ) -> None:
        """Atomically restore a snapshot captured by :meth:`snapshot`."""
        restored: dict[str, tuple[TerminalBackendDefinition, str]] = {}
        for name, definition, owner in snapshot:
            if not isinstance(definition, TerminalBackendDefinition):
                raise TypeError("Invalid terminal backend registry snapshot")
            if name != definition.name or not str(owner or "").strip():
                raise ValueError("Invalid terminal backend registry snapshot")
            restored[name] = (replace(definition), owner)
        with self._lock:
            self._definitions = restored

    def snapshot_owner(
        self,
        owner: str,
    ) -> tuple[tuple[str, TerminalBackendDefinition, str], ...]:
        """Capture registrations for one plugin before calling register()."""
        owner = str(owner or "").strip()
        return tuple(entry for entry in self.snapshot() if entry[2] == owner)

    def restore_owner(
        self,
        owner: str,
        snapshot: tuple[tuple[str, TerminalBackendDefinition, str], ...],
    ) -> None:
        """Roll one plugin back without disturbing registrations from others."""
        owner = str(owner or "").strip()
        restored: dict[str, tuple[TerminalBackendDefinition, str]] = {}
        for name, definition, registered_owner in snapshot:
            if not isinstance(definition, TerminalBackendDefinition):
                raise TypeError("Invalid terminal backend owner snapshot")
            if registered_owner != owner or name != definition.name:
                raise ValueError("Terminal backend owner snapshot mismatch")
            restored[name] = (replace(definition), registered_owner)

        with self._lock:
            for name in tuple(self._definitions):
                if self._definitions[name][1] == owner:
                    self._definitions.pop(name)
            self._definitions.update(restored)

    def unregister(self, name: str, *, owner: str | None = None) -> bool:
        """Remove one backend when it still belongs to *owner*."""
        name = str(name).lower()
        owner = str(owner or "").strip() or None
        with self._lock:
            entry = self._definitions.get(name)
            if entry is None:
                return False
            if owner is not None and entry[1] != owner:
                return False
            self._definitions.pop(name, None)
            return True

    def unregister_plugin_backends(self) -> None:
        """Clear plugin registrations before a forced plugin rediscovery."""
        with self._lock:
            self._definitions.clear()


terminal_backend_registry = TerminalBackendRegistry()
