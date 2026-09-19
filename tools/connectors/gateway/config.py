"""Configuration and availability gate for connector tools.

Availability fails closed; the gateway remains authoritative for entitlement and route availability.
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

# Context cap, not a wire limit; the gateway batch cap is deliberately unreachable.
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
        from hermes_cli.config import load_config_readonly as _load

        cfg = _load() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            tools_cfg = {}
        return ConnectorConfig.from_raw(tools_cfg.get("connectors"))
    except Exception as e:
        logger.debug("Failed to load connector config: %s", e)
        return ConnectorConfig.from_raw(None)


def managed_tools_rolled_out() -> bool:
    """The portal has enabled connectors for this account.

    The gateway answers 404 to every ``/v1/connectors`` route for an account the portal has not
    enabled, and 404 is indistinguishable from "dark" by design. Paid access or a free tool pool
    says nothing about that, so entitlement is the wrong predicate here: the portal mints its
    answer onto the token as ``managed_tools`` and this reads only that. A token minted before
    the claim existed carries none and reads as not enabled."""
    from hermes_cli.nous_account import get_nous_portal_account_info

    account_info = get_nous_portal_account_info()
    return bool(account_info.logged_in) and account_info.managed_tools_rolled_out


def connectors_available(
    config_loader: Optional[Callable[[], ConnectorConfig]] = None,
    entitlement_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Fail closed so availability failures do not become model-visible errors.

    The one gate for the connectors surface, and the tool's ``check_fn``: outside it the tool is
    not in the schema at all, so the model never narrates a gateway 404 to a user the portal has
    not enabled. Free-tier identities are always in; accounts are in only when the portal says so
    via the token claim."""
    try:
        resolved_loader = config_loader or load_config
        if not resolved_loader().enabled:
            return False
        if entitlement_check is None:
            from hermes_cli.anon_auth import is_guest_state
            from tools.managed_tool_gateway import _read_nous_provider_state

            # Availability must not mint or refresh an identity.
            if is_guest_state(_read_nous_provider_state()):
                return True

            entitlement_check = managed_tools_rolled_out
        return bool(entitlement_check())
    except Exception as e:
        logger.debug("Connector availability check failed: %s", e)
        return False


def operation_session_key(session_id: Optional[str]) -> str:
    """The key an operation is registered under: the gateway session key the RPCs look up by
    (``HERMES_SESSION_KEY``), falling back to the agent's session id where no gateway bound one.
    The agent id alone is wrong on the desktop: compaction rotates it mid-turn while the gateway
    key stays, and a card keyed by the old id can no longer be driven."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "") or str(session_id or "")


def session_platform() -> str:
    """The session's surface (``desktop``, ``tui``, ``cli``, a messaging platform, or '').

    Decides whether a card exists for this session. Never infer that from a callback being
    attached: the GUI bridge attaches callbacks to every backend session, terminal TUI included.
    Messaging adapters bind the surface as the session platform; the desktop and TUI gateway bind
    it as the session source (``tui_gateway.server._set_session_context``), so both are read."""
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "") or get_session_env("HERMES_SESSION_SOURCE", "")
    return str(platform or "").strip().lower()
