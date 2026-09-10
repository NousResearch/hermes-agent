"""Configuration gate for connector tools.

Mirrors the ``ToolSearchConfig`` idiom in ``tools/tool_search.py``: a frozen
dataclass built by a tolerant ``from_raw`` so a typo in user config degrades
to defaults instead of breaking the agent.

Availability is a two-leg AND that fails closed:
    connectors_available() = config flag AND managed_nous_tools_enabled()
The config flag is the user's off switch; the entitlement leg is the portal
sign-in every managed tool already gates on. The gateway remains authoritative:
404 routes degrade to local-only, and execution refusals reach the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CALLS_PER_DISPATCH",
    "ConnectorConfig",
    "connectors_available",
    "load_config",
]

# Bound the connector entries one tool_call dispatch may carry — the same
# constant family as tool_search's _MAX_QUERIES_PER_CALL. Context management,
# not a wire limit: the gateway's own batch cap (25) is unreachable from
# here by design, so there is no chunking code anywhere.
MAX_CALLS_PER_DISPATCH = 10

_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ConnectorConfig:
    """Resolved ``tools.connectors`` configuration."""

    enabled: bool = True

    @classmethod
    def from_raw(cls, raw: Any) -> "ConnectorConfig":
        """Build a config from a raw dict / bool / None.

        Tolerant by design: unknown shapes and garbage values fall back to
        the default (enabled) rather than raising. The effective gate for
        signed-out users is the entitlement leg, not this flag.
        """
        if isinstance(raw, bool):
            return cls(enabled=raw)
        if isinstance(raw, dict):
            return cls(enabled=_coerce_bool(raw.get("enabled"), True))
        return cls()


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return fallback


def load_config() -> ConnectorConfig:
    """Load connector config from the user config file."""
    try:
        from hermes_cli.config import load_config_readonly as _load

        cfg = _load() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            tools_cfg = {}
        return ConnectorConfig.from_raw(tools_cfg.get("connectors"))
    except Exception as e:
        logger.debug("Failed to load connector config: %s", e)
        return ConnectorConfig.from_raw(None)


def connectors_available(
    config_loader: Optional[Callable[[], ConnectorConfig]] = None,
    entitlement_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """True when connector routes may be attempted at all. Fails closed.

    Any exception in either leg counts as unavailable — this function is on
    the tool_search availability path, where a connector problem must never
    become a model-visible error.
    """
    try:
        resolved_loader = config_loader or load_config
        if not resolved_loader().enabled:
            return False
        if entitlement_check is None:
            from tools.tool_backend_helpers import managed_nous_tools_enabled

            entitlement_check = managed_nous_tools_enabled
        return bool(entitlement_check())
    except Exception as e:
        logger.debug("Connector availability check failed: %s", e)
        return False
