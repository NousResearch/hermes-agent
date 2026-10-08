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
    "ensure_guest_identity",
    "guest_identity_pending",
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
    not enabled. Free-tier identities are always in, and so is a user with no identity who may get
    one; accounts are in only when the portal says so via the token claim."""
    try:
        resolved_loader = config_loader or load_config
        if not resolved_loader().enabled:
            return False
        if entitlement_check is None:
            from hermes_cli.anon_auth import is_guest_state
            from tools.managed_tool_gateway import _read_nous_provider_state

            # Availability must not mint or refresh an identity.
            if is_guest_state(_read_nous_provider_state()) or guest_identity_pending():
                return True

            entitlement_check = managed_tools_rolled_out
        return bool(entitlement_check())
    except Exception as e:
        logger.debug("Connector availability check failed: %s", e)
        return False


def guest_identity_pending() -> bool:
    """No Nous identity or bearer yet, and a guest one may be created (``nous.guest`` is not false
    and no earlier mint was refused for good). No network: this feeds the ``check_fn``."""
    from hermes_cli.anon_auth import guest_allowed, last_mint_failure
    from tools.managed_tool_gateway import _read_nous_provider_state, peek_nous_access_token

    if not guest_allowed() or _read_nous_provider_state() is not None or peek_nous_access_token():
        return False
    failure = last_mint_failure()
    return not failure or bool(failure["retryable"])


_SETUP_FAILED = "Hosted connectors could not be set up. "


def guest_setup_failure() -> Optional[str]:
    """The last failed guest creation for this profile, worded with the wait still left, or None."""
    from hermes_cli.anon_auth import anon_failure_copy, last_mint_failure

    if not (failure := last_mint_failure()):
        return None
    return _SETUP_FAILED + anon_failure_copy(failure["error_code"], retry_after=failure["retry_after"])


def ensure_guest_identity() -> Optional[str]:
    """Create the guest identity on the first hosted connector action, synchronously.

    Returns None when an identity exists or was just created, else the failure copy for the model:
    a refusal, a rate limit or an unreachable portal, with the wait when there is one."""
    if not guest_identity_pending():
        return None
    from hermes_cli.anon_auth import ensure_portal_identity
    from hermes_cli.auth_constants import AuthError

    try:
        if ensure_portal_identity(explicit=True, for_connectors=True) is not None:
            return None
    except AuthError as exc:
        return _SETUP_FAILED + str(exc)
    return guest_setup_failure() or _SETUP_FAILED + "Try again shortly."


def operation_session_key(session_id: Optional[str]) -> str:
    """The key an operation is registered under: the gateway session key the RPCs look up by
    (``HERMES_SESSION_KEY``), falling back to the agent's session id where no gateway bound one.
    The agent id alone is wrong on the desktop: compaction rotates it mid-turn while the gateway
    key stays, and a card keyed by the old id can no longer be driven."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "") or str(session_id or "")


def session_platform() -> str:
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "") or get_session_env("HERMES_SESSION_SOURCE", "")
    return str(platform or "").strip().lower()
