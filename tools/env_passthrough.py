"""Environment variable passthrough registry: the session-scoped allowlist of vars a
skill's ``required_environment_variables`` (registered by ``skill_view``) or
``terminal.env_passthrough`` in config.yaml may forward into sandboxed children
(execute_code, terminal), which strip secrets by default. Under profile multiplexing,
forwarded values resolve through the profile's secret scope, not the process env."""

from __future__ import annotations

import logging
import os
from typing import Iterable
from hermes_cli.config import cfg_get, read_raw_config

logger = logging.getLogger(__name__)

# Process-wide set of env var names registered by skills for sandbox
# passthrough. Deliberately NOT a ContextVar: tool dispatch fans each tool
# call onto a worker whose context is a copy_context() snapshot taken at
# submit time (tools.thread_context.propagate_context_to_thread), so a
# registration made inside one tool's worker (skill_view calling
# register_env_passthrough) never reaches the submitting thread's context —
# every subsequent tool (execute_code, terminal) re-snapshots the original
# context and sees an empty allowlist, and the skill's declared env vars
# never pass through (#90004). The config-based allowlist below is already
# a module-level global with exactly the process-wide visibility the skill
# path needs to match.
#
# Cross-session exposure is limited to the NAMES: the values still resolve
# per profile through resolve_passthrough_value's secret_scope, so a name
# registered by one session cannot read another profile's secret. The
# previous ContextVar also never actually isolated anything within one
# process running a single profile (the common deployment).
_allowed_env_vars: set[str] = set()


def _get_allowed() -> set[str]:
    """Get the process-wide skill passthrough allowlist."""
    return _allowed_env_vars


# Config-based allowlist, keyed by Hermes home: under gateway.multiplex_profiles one process serves
# many profiles, and a single slot would let the first profile's operator allowlist decide which env
# vars tunnel into every other profile's sandbox children.
_config_passthrough: dict[str, frozenset[str]] = {}


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
            _is_hermes_internal_secret, _is_provider_env_blocklisted)
    except Exception as e:
        logger.warning(
            "env passthrough: provider credential blocklist import failed; "
            "failing closed and refusing passthrough registration for %r: %s", name, e)
        return True
    # Case-folded membership too: the remote-exec env builder resolves each
    # registered name via os.getenv(), which is case-insensitive on Windows, so
    # ``openai_api_key`` would tunnel the real OPENAI_API_KEY into children.
    return _is_hermes_internal_secret(name) or _is_provider_env_blocklisted(name)


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
    from hermes_constants import hermes_home_key

    try:
        home_key = hermes_home_key()
    except (RuntimeError, OSError):
        # No resolvable home (stripped environ in a sandbox child): nothing to scope by.
        home_key = ""
    cached = _config_passthrough.get(home_key)
    if cached is not None:
        return cached
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
    _config_passthrough[home_key] = frozenset(result)
    return _config_passthrough[home_key]


def is_env_passthrough(var_name: str) -> bool:
    """True if *var_name* was registered by a skill or listed in config and is not a
    Hermes-managed credential NOW. Ownership changes after acceptance (a platform plugin
    registered later declares the name in its ``required_env`` or manifest), so the refusal applied at registration
    is re-applied here, where every child builder consumes the allowlist."""
    return ((var_name in _get_allowed() or var_name in _load_config_passthrough())
            and not _is_hermes_provider_credential(var_name))


def get_all_passthrough() -> frozenset[str]:
    """Return the union of skill-registered and config-based passthrough vars, minus names
    that have become Hermes-managed credentials since they were accepted."""
    return frozenset(name for name in frozenset(_get_allowed()) | _load_config_passthrough()
                     if not _is_hermes_provider_credential(name))


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


def scoped_passthrough_additions(present: Iterable[str]) -> dict[str, str]:
    """Declared passthrough names the bound profile secret scope supplies but the env being
    filtered (*present*) lacks. A routed profile's ``.env`` and hydrated sources never enter
    ``os.environ`` (``load_hermes_dotenv`` skips the process-global load for a routed home), so a
    name-by-name filter over the process env can only forward a declared name the LAUNCH profile
    also happens to define — the served profile's own value has no way in (#114209). Reads the
    bound scope alone: never ``os.environ``, never another profile. Empty without a scope, so
    single-profile spawns are byte-identical."""
    from agent.secret_scope import _is_global_env, current_secret_scope
    scope = current_secret_scope()
    if not scope:
        return {}
    present = set(present)
    additions: dict[str, str] = {}
    for name in get_all_passthrough():
        if name in present or _is_global_env(name):
            continue
        value = scope.get(name)
        if value is not None:
            additions[name] = value
    return additions


def routed_profile_passthrough_additions(
    hermes_home: str | os.PathLike, names: Iterable[str]
) -> dict[str, str]:
    """Of *names* (just stripped as the launch profile's residue), the ones the ROUTED
    profile *hermes_home* declares in ``terminal.env_passthrough`` and supplies itself.

    ``strip_launch_profile_env`` drops every name the launch profile's external sources
    supplied, on the assumption that a caller's bound secret scope puts the target's own
    values back. Outside the gateway/multiplex path nothing binds that scope — a
    single-process worker acting for a routed home (desktop backend, ``key_cmd`` helper,
    a worker spawned with an explicit target home) — so nothing put them back and a
    declared passthrough name that lived only in the target's vault vanished from the
    target's terminal children, while the same name reached them under a gateway-served
    turn. The verification surface then reported a working credential as missing.

    Resolves the target profile alone: hydrate ITS sources, build ITS scope, read ITS
    ``terminal.env_passthrough`` (the home override is bound for the read so a caller that
    passed ``target_home`` explicitly without an override still gets the target's list).
    Never ``os.environ``, never another profile. Only names already removed are candidates,
    so a child that never had a value does not gain one. A bound scope wins: the
    multiplex/gateway path already overlays it through :func:`scoped_passthrough_additions`,
    and an unscoped task under multiplex semantics must stay fail-closed rather than
    quietly read files.
    """
    from agent.secret_scope import (
        _is_global_env, build_profile_secret_scope, current_secret_scope, is_multiplex_active)
    if current_secret_scope() is not None or is_multiplex_active():
        return {}
    target = str(hermes_home or "")
    if not target:
        return {}
    stripped = set(names)
    if not stripped:
        return {}
    from pathlib import Path

    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import (
        get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override)

    token = None
    if str(get_hermes_home_override() or "") != target:
        token = set_hermes_home_override(target)
    try:
        hydrate_profile_secret_sources(target)
        scope = build_profile_secret_scope(Path(target))
        declared = get_all_passthrough()
    finally:
        if token is not None:
            reset_hermes_home_override(token)
    additions: dict[str, str] = {}
    for name in declared:
        if name not in stripped or _is_global_env(name):
            continue
        value = scope.get(name)
        if value is not None:
            additions[name] = value
    return additions


def clear_env_passthrough() -> None:
    """Reset the skill-registered allowlist (e.g. on session reset).

    Clears the process-wide set; a later ``skill_view`` re-registers its
    vars on demand, so recovery is a single skill load."""
    _get_allowed().clear()
