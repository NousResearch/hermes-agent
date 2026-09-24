"""Environment variable passthrough registry: the session-scoped allowlist of vars a
skill's ``required_environment_variables`` (registered by ``skill_view``) or
``terminal.env_passthrough`` in config.yaml may forward into sandboxed children
(execute_code, terminal), which strip secrets by default. Under profile multiplexing,
forwarded values resolve through the profile's secret scope, not the process env."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Iterable
from hermes_cli.config import cfg_get, read_raw_config

logger = logging.getLogger(__name__)

# Session-scoped allowlist; ContextVar-backed to prevent cross-session bleed
# in the gateway pipeline.
_allowed_env_vars_var: ContextVar[set[str]] = ContextVar("_allowed_env_vars")


def _get_allowed() -> set[str]:
    """Get or create the allowed env vars set for the current context/session."""
    try:
        return _allowed_env_vars_var.get()
    except LookupError:
        val: set[str] = set()
        _allowed_env_vars_var.set(val)
        return val


# Cache for the config-based allowlist (loaded once per process).
_config_passthrough: frozenset[str] | None = None


def _is_hermes_provider_credential(name: str) -> bool:
    """True if ``name`` is a Hermes-managed provider credential per
    ``_HERMES_PROVIDER_ENV_BLOCKLIST`` or a dynamic Hermes-internal secret
    (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_*). Skill-declared
    ``required_environment_variables`` must not override this — that was the
    GHSA-rhgp-j443-p4rf bypass (a skill registered ``OPENAI_API_KEY`` and received it
    in the ``execute_code`` child); non-Hermes keys (TENOR_API_KEY, …) stay
    registerable. Fails closed when the blocklist cannot be imported."""
    try:
        from tools.environments.local_env_policy import (
            _HERMES_PROVIDER_ENV_BLOCKLIST, _is_hermes_internal_secret)
    except Exception as e:
        logger.warning(
            "env passthrough: provider credential blocklist import failed; "
            "failing closed and refusing passthrough registration for %r: %s", name, e)
        return True
    return _is_hermes_internal_secret(name) or name in _HERMES_PROVIDER_ENV_BLOCKLIST


def register_env_passthrough(var_names: Iterable[str]) -> None:
    """Register env var names as allowed in sandboxed environments (typically a
    skill's ``required_environment_variables``). Hermes-managed provider credentials
    are rejected (GHSA-rhgp-j443-p4rf) — such skills should use the main-process tools
    (web_search, web_extract, …); third-party keys pass normally."""
    for name in _accepted((n.strip() for n in var_names), (
        "env passthrough: refusing to register Hermes provider "
        "credential %r (blocked by _HERMES_PROVIDER_ENV_BLOCKLIST). "
        "Skills must not override the execute_code sandbox's "
        "credential scrubbing; see GHSA-rhgp-j443-p4rf."
    )):
        _get_allowed().add(name)
        logger.debug("env passthrough: registered %s", name)


def _accepted(names, refusal_msg: str):
    """Yield non-empty *names* that are not Hermes provider credentials; refused
    names are logged with *refusal_msg* (``%r`` = name)."""
    for name in names:
        if not name:
            continue
        if _is_hermes_provider_credential(name):
            logger.warning(refusal_msg, name)
            continue
        yield name


def _load_config_passthrough() -> frozenset[str]:
    """Load ``tools.env_passthrough`` from config.yaml (cached). Same credential
    filter as register_env_passthrough: operator config must not tunnel provider
    credentials into sandbox children either (GHSA-rhgp-j443-p4rf)."""
    global _config_passthrough
    if _config_passthrough is not None:
        return _config_passthrough
    result: set[str] = set()
    try:
        passthrough = cfg_get(read_raw_config(), "terminal", "env_passthrough")
        items = passthrough if isinstance(passthrough, list) else ()
        result.update(_accepted((i.strip() for i in items if isinstance(i, str)), (
            "env passthrough: refusing to register Hermes "
            "provider credential %r from config.yaml (blocked "
            "by _HERMES_PROVIDER_ENV_BLOCKLIST). Operator "
            "configuration must not override the execute_code "
            "sandbox's credential scrubbing; see "
            "GHSA-rhgp-j443-p4rf."
        )))
    except Exception as e:
        logger.debug("Could not read tools.env_passthrough from config: %s", e)
    _config_passthrough = frozenset(result)
    return _config_passthrough


def is_env_passthrough(var_name: str) -> bool:
    """True if *var_name* was registered by a skill or listed in config."""
    return var_name in _get_allowed() or var_name in _load_config_passthrough()


def get_all_passthrough() -> frozenset[str]:
    """Return the union of skill-registered and config-based passthrough vars."""
    return frozenset(_get_allowed()) | _load_config_passthrough()


def resolve_passthrough_value(name: str, fallback: str | None = None) -> str | None:
    """Resolve an allowlisted variable without crossing profile boundaries. ``fallback``
    is what the caller would have forwarded before secret scopes existed (a snapshot of
    ``os.environ`` / the profile ``.env``). An active multiplex scope is authoritative:
    a missing key returns ``None``, never the process-global env, and an unscoped read
    raises the fail-closed ``UnscopedSecretError``. Outside multiplexing an installed
    scope keeps overlay semantics and an unscoped caller keeps its fallback."""
    from agent.secret_scope import (
        _is_global_env, current_secret_scope, get_secret, is_multiplex_active)
    # Global terminal/runtime settings are not profile secrets; ``fallback`` is
    # already the caller's effective value (incl. an explicit per-call override).
    if _is_global_env(name) and fallback is not None:
        return fallback
    multiplex_active = is_multiplex_active()
    if current_secret_scope() is None:
        return get_secret(name) if multiplex_active else fallback
    return get_secret(name, None if multiplex_active else fallback)


def clear_env_passthrough() -> None:
    """Reset the skill-scoped allowlist (e.g. on session reset)."""
    _get_allowed().clear()
