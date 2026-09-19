"""Extensible action contract for the canonical Discord tools.

Feature modules under :mod:`tools.discord_api` may register bounded actions
without registering another public model tool or growing ``tools/discord_tool.py``
with feature-specific code.  The canonical ``discord`` / ``discord_admin``
owners discover ``*_action.py`` modules, merge their schemas and policy
metadata, and retain token, allowlist, transport, and error ownership.
"""

from __future__ import annotations

import importlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DiscordActionSurface = Literal["core", "admin"]
DiscordActionHandler = Callable[..., str]

_ACTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTIONS: Dict[str, "DiscordAction"] = {}
_DISCOVERED_MODULES: set[str] = set()
_DISCOVERY_LOCK = threading.RLock()


@dataclass(frozen=True)
class DiscordAction:
    """One action contributed to a canonical Discord model tool.

    ``handler`` receives the active profile token, the canonical REST request
    callable as ``_request``, the built-in Discord arguments, and every
    action-specific property declared in ``properties``.
    """

    name: str
    surface: DiscordActionSurface
    signature: str
    description: str
    handler: DiscordActionHandler
    properties: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    required: Tuple[str, ...] = ()
    permission_hint: Optional[str] = None
    members_intent_required: bool = False

    def __post_init__(self) -> None:
        if not _ACTION_NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid Discord action name: {self.name!r}")
        if self.surface not in {"core", "admin"}:
            raise ValueError(f"invalid Discord action surface: {self.surface!r}")
        if not callable(self.handler):
            raise TypeError(f"handler for {self.name!r} must be callable")
        if not isinstance(self.signature, str) or not self.signature.startswith("("):
            raise ValueError(f"signature for {self.name!r} must be parenthesized")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"description for {self.name!r} must be non-empty")
        if "action" in self.properties:
            raise ValueError("extension actions cannot replace the canonical 'action' field")
        for property_name, schema in self.properties.items():
            if not _ACTION_NAME_RE.fullmatch(property_name):
                raise ValueError(
                    f"invalid property name for {self.name!r}: {property_name!r}"
                )
            if not isinstance(schema, Mapping):
                raise TypeError(
                    f"schema for {self.name!r}.{property_name} must be a mapping"
                )
        if len(set(self.required)) != len(self.required):
            raise ValueError(f"duplicate required property on {self.name!r}")
        missing = set(self.required) - set(self.properties)
        if missing:
            raise ValueError(
                f"required properties missing from {self.name!r} schema: "
                f"{', '.join(sorted(missing))}"
            )


def register_discord_action(action: DiscordAction) -> DiscordAction:
    """Register one bounded action, rejecting ambiguous ownership."""
    with _DISCOVERY_LOCK:
        existing = _ACTIONS.get(action.name)
        if existing is not None and existing is not action:
            raise ValueError(f"Discord action already registered: {action.name}")
        _ACTIONS[action.name] = action
    return action


def discover_discord_actions(package_dir: Optional[Path] = None) -> Tuple[str, ...]:
    """Import every ``*_action.py`` module in deterministic order.

    A broken optional action is isolated: it is logged and skipped rather than
    taking both canonical Discord tools out of the process.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    imported: list[str] = []
    with _DISCOVERY_LOCK:
        for path in sorted(root.glob("*_action.py")):
            module_name = f"{__package__}.{path.stem}"
            if module_name in _DISCOVERED_MODULES:
                continue
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - defensive isolation
                logger.warning("Could not import Discord action module %s: %s", module_name, exc)
                continue
            _DISCOVERED_MODULES.add(module_name)
            imported.append(module_name)
    return tuple(imported)


def get_discord_actions(
    surface: Optional[DiscordActionSurface] = None,
) -> Dict[str, DiscordAction]:
    """Return a registration-order snapshot, optionally filtered by surface."""
    with _DISCOVERY_LOCK:
        if surface is None:
            return dict(_ACTIONS)
        return {name: action for name, action in _ACTIONS.items() if action.surface == surface}
