"""Per-turn terminal scope: profile-scoped TERMINAL_* policy.

Multiplexed surfaces (gateway, dashboard/TUI, cron) serve several profiles from one process;
mirroring terminal settings into ``os.environ`` let the first profile pin its backend onto
everyone else (sandbox escape). Like ``agent/secret_scope.py``, a ContextVar holds the active
profile's COMPLETE ``TERMINAL_*`` policy; while bound, ``terminal_env`` resolves ONLY from it
(omitted keys -> defined default, never ambient env). If the policy cannot be resolved a
*refusal* scope is installed and terminal execution raises :class:`TerminalPolicyUnavailable`.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# None = no scope bound (process-env behavior); dict = complete policy; Refusal = resolution failed.
_terminal_scope_var: ContextVar = ContextVar("hermes_terminal_scope", default=None)

# Keys whose default lives in terminal_tool.py, not DEFAULT_CONFIG (which wins on overlap);
# without them the projection is not total.
_TOOL_LEVEL_DEFAULTS: Dict[str, Any] = {
    "cwd": ".", "ssh_host": "", "ssh_user": "", "ssh_port": 22, "ssh_key": "",
    "docker_orphan_reaper": True, "docker_persist_across_processes": True,
    "sandbox_dir": "", "lifetime_seconds": 300, "docker_shared_container_key": "",
    "home_mode": "auto",
}


class TerminalPolicyUnavailable(Exception):
    """The routed profile's ``.env``/``config.yaml`` exists but cannot be read/parsed."""


class TerminalPolicyRefusal(Dict[str, str]):
    """Marker scope (empty dict subclass) installed when policy resolution failed."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


def set_terminal_scope(mapping: Optional[Dict[str, str]]) -> Token:
    """Install *mapping* as the current context's terminal policy."""
    return _terminal_scope_var.set(mapping)


def reset_terminal_scope(token: Token) -> None:
    _terminal_scope_var.reset(token)


def get_terminal_scope() -> Optional[Dict[str, str]]:
    """The active scope mapping/refusal, or ``None`` when no scope is bound."""
    return _terminal_scope_var.get()


def enforce_no_refusal() -> None:
    """Raise when the active scope is a refusal scope (fail closed).

    Execution paths (terminal tool, execute_code) call this before spawning anything: under a refusal scope
    the profile's terminal policy could not be resolved, and running with the launch process's ambient
    policy is exactly the authority leak this module closes (#68559 requires refusal, not fallback).
    Non-scoped and policy-scoped contexts pass silently.
    """
    scope = _terminal_scope_var.get()
    if isinstance(scope, TerminalPolicyRefusal):
        raise TerminalPolicyUnavailable(
            f"terminal policy unavailable for this profile: {scope.reason}")


def terminal_env(name: str, default: str = "") -> str:
    """Authoritative read of a ``TERMINAL_*`` variable.

    No scope: process env, then *default*. Refusal scope: raise. Policy scope: ONLY the
    policy; a missing key yields *default*, never os.environ.
    """
    scope = _terminal_scope_var.get()
    if scope is None:
        return os.environ.get(name, default)
    enforce_no_refusal()
    value = scope.get(name)
    return default if value is None else str(value)


def build_profile_terminal_scope(hermes_home: "Any") -> Dict[str, str]:
    """Build the COMPLETE effective ``TERMINAL_*`` policy for a profile home.

    Projection: ``DEFAULT_CONFIG['terminal']`` <- profile ``.env`` TERMINAL_* <- profile
    ``config.yaml`` ``terminal:``. Total by construction, so a bound scope never widens back to
    ambient authority. Raises :class:`TerminalPolicyUnavailable` if a present file is unreadable.
    """
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    home = Path(hermes_home)
    scope: Dict[str, str] = {}

    def _apply(mapping: Dict[str, Any]) -> None:
        for cfg_key, value in mapping.items():
            # cwd placeholders are resolved per-surface later; not a policy value.
            if value is None or (cfg_key == "cwd" and str(value).strip() in {".", "auto", "cwd"}):
                continue
            env_var = TERMINAL_CONFIG_ENV_MAP.get(cfg_key)
            if env_var:
                scope[env_var] = str(value)

    _apply({**_TOOL_LEVEL_DEFAULTS, **(DEFAULT_CONFIG.get("terminal") or {})})
    env_path = home / ".env"
    if env_path.exists():
        # load_env_file swallows OSError by design (secret scope fails soft); an unreadable
        # profile .env must fail closed here.
        try:
            env_path.read_bytes()
        except Exception as exc:
            raise TerminalPolicyUnavailable(f"cannot read {env_path}: {exc}") from exc
        from agent.secret_scope import load_env_file

        scope.update((k, str(v)) for k, v in load_env_file(env_path).items()
                     if k.startswith("TERMINAL_"))
    # Read config.yaml directly, not via read_raw_config() (which collapses "missing" and
    # "unparseable" into {}): present-but-unparseable must fail closed.
    config_path = home / "config.yaml"
    try:
        config_exists = config_path.exists()
    except Exception as exc:
        raise TerminalPolicyUnavailable(f"cannot resolve terminal config in {home}: {exc}") from exc
    if config_exists:
        from hermes_cli.config import fast_safe_load

        try:
            with open(config_path, encoding="utf-8") as f:
                raw = fast_safe_load(f)
        except Exception as exc:
            raise TerminalPolicyUnavailable(f"cannot parse {config_path}: {exc}") from exc
        raw_terminal = raw.get("terminal") if isinstance(raw, dict) else None
        if isinstance(raw_terminal, dict):
            _apply(raw_terminal)
    return scope


def install_profile_terminal_scope(hermes_home: "Any") -> Token:
    """Build AND install a profile's policy; on failure install the refusal scope. Never raises."""
    try:
        return set_terminal_scope(build_profile_terminal_scope(hermes_home))
    except TerminalPolicyUnavailable as exc:
        logger.warning("terminal policy unavailable: %s", exc)
        return _terminal_scope_var.set(TerminalPolicyRefusal(str(exc)))


@contextmanager
def install_and_reset_profile_terminal_scope(hermes_home: "Any") -> Iterator[None]:
    """Install the profile's terminal policy for a bounded turn/fire. Never raises."""
    token = install_profile_terminal_scope(hermes_home)
    try:
        yield
    finally:
        reset_terminal_scope(token)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def install_refusal_scope(reason: str) -> Token:
    """Install a refusal scope after :class:`TerminalPolicyUnavailable`.

    Terminal execution under this scope is rejected (fail closed) instead of
    running under the launch process's ambient policy.
    """
    return _terminal_scope_var.set(TerminalPolicyRefusal(reason))

@contextmanager
def terminal_scope(mapping: Optional[Dict[str, str]]) -> Iterator[None]:
    """Context manager form of set/reset_terminal_scope."""
    token = set_terminal_scope(mapping)
    try:
        yield
    finally:
        reset_terminal_scope(token)
# ---- END PLUGIN-COMPAT ----
