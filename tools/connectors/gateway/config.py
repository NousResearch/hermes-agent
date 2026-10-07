"""Configuration and availability gate for connector tools.

The hosted tool gateway is not part of this build; only local MCP servers remain.
Availability fails closed on malformed configuration.
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

# Context cap, not a wire limit.
MAX_CALLS_PER_DISPATCH = 10

_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ConnectorConfig:

    enabled: bool = True

    @classmethod
    def from_raw(cls, raw: Any) -> "ConnectorConfig":
        """Malformed configuration falls back to the enabled default."""
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
    try:
        from rabbit_cli.config import load_config_readonly as _load

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
    """Fail closed so availability failures do not become model-visible errors.

    The one gate for the connectors surface, and the tool's ``check_fn``. There is no
    hosted entitlement left to check: local MCP servers are available whenever the
    config flag allows them."""
    try:
        resolved_loader = config_loader or load_config
        if not resolved_loader().enabled:
            return False
        if entitlement_check is not None:
            return bool(entitlement_check())
        return True
    except Exception as e:
        logger.debug("Connector availability check failed: %s", e)
        return False


def operation_session_key(session_id: Optional[str]) -> str:
    """The key an operation is registered under: the gateway session key the RPCs look up by
    (``RABBIT_SESSION_KEY``), falling back to the agent's session id where no gateway bound one."""
    from gateway.session_context import get_session_env

    return get_session_env("RABBIT_SESSION_KEY", "") or str(session_id or "")


def session_platform() -> str:
    from gateway.session_context import get_session_env

    platform = get_session_env("RABBIT_SESSION_PLATFORM", "") or get_session_env("RABBIT_SESSION_SOURCE", "")
    return str(platform or "").strip().lower()
