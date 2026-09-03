"""Per-turn terminal scope: profile-scoped TERMINAL_* policy.

The multiplexing gateway (and the unified dashboard/TUI, and cron) serve
several Hermes profiles from one process. Terminal settings were historically
mirrored into the process-global ``os.environ`` (first writer wins), so the
first profile to touch the terminal after startup pinned its backend — and
every other setting — onto all later turns: a ``local`` profile silently
executing inside another profile's docker sandbox, or the reverse (a sandbox
escape). Mirrors the isolation seam that ``agent/secret_scope.py`` provides
for credentials: a ContextVar holds the active profile's COMPLETE effective
``TERMINAL_*`` policy, installed at each in-process profile boundary.

Two contracts distinguish this from a plain override dict:

- **Authoritative projection.** While a scope is bound, ``terminal_env``
  resolves ONLY from that policy (built from defined defaults + the profile's
  ``.env`` + its ``config.yaml`` explicit keys). Omitted keys resolve to the
  defined default — never to ambient ``os.environ`` — so a routed profile can
  neither inherit nor be escaped onto the launch process's mounts, SSH
  targets, or resource policy (#68559).
- **Fail closed.** If the profile's policy cannot be resolved (unreadable or
  malformed ``.env``/``config.yaml``), the install raises
  :class:`TerminalPolicyUnavailable` and callers must install a *refusal*
  scope; terminal execution under a refusal scope is rejected outright
  rather than falling back to ambient authority.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# ``None`` = no scope bound in this context; readers use the historical
# process-env behavior (single-process CLI/TUI, unaffected surfaces).
# A dict = the active profile's complete effective terminal policy.
# A TerminalPolicyRefusal = resolution failed; terminal execution must refuse.
_terminal_scope_var: ContextVar = ContextVar("hermes_terminal_scope", default=None)


class TerminalPolicyUnavailable(Exception):
    """The routed profile's terminal policy could not be resolved.

    Raised when the profile's ``.env`` or ``config.yaml`` exists but cannot be
    read/parsed. Callers must install the returned refusal scope instead of
    continuing without a scope — executing under ambient process authority is
    exactly the leak this module exists to close.
    """


class TerminalPolicyRefusal(Dict[str, str]):
    """Marker scope installed when policy resolution failed.

    An (empty) dict subclass so existing dict-typed checks keep working, with
    a flag that makes ``terminal_env`` raise before any value is served.
    """

    refused = True

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


def set_terminal_scope(mapping: Optional[Dict[str, str]]) -> Token:
    """Install *mapping* as the current context's terminal policy."""
    return _terminal_scope_var.set(mapping)


def install_refusal_scope(reason: str) -> Token:
    """Install a refusal scope after :class:`TerminalPolicyUnavailable`.

    Terminal execution under this scope is rejected (fail closed) instead of
    running under the launch process's ambient policy.
    """
    return _terminal_scope_var.set(TerminalPolicyRefusal(reason))


def reset_terminal_scope(token: Token) -> None:
    _terminal_scope_var.reset(token)


def get_terminal_scope() -> Optional[Dict[str, str]]:
    """The active scope mapping/refusal, or ``None`` when no scope is bound."""
    return _terminal_scope_var.get()


@contextmanager
def terminal_scope(mapping: Optional[Dict[str, str]]) -> Iterator[None]:
    """Context manager form of set/reset_terminal_scope."""
    token = set_terminal_scope(mapping)
    try:
        yield
    finally:
        reset_terminal_scope(token)


def terminal_env(name: str, default: str = "") -> str:
    """Authoritative read of a ``TERMINAL_*`` variable.

    - No scope bound: process env, then *default* (historical single-process
      behavior — CLI/TUI surfaces that never route profiles are unchanged).
    - Refusal scope bound: raise — policy is unavailable and execution must
      fail closed, not fall back to ambient authority.
    - Policy scope bound: resolve ONLY from the policy; a missing key yields
      the *default* (which callers derive from defined defaults), never
      ``os.environ``.
    """
    scope = _terminal_scope_var.get()
    if scope is None:
        import os

        return os.environ.get(name, default)
    if isinstance(scope, TerminalPolicyRefusal):
        raise TerminalPolicyUnavailable(
            f"terminal policy unavailable for this profile: {scope.reason}"
        )
    value = scope.get(name)
    if value is not None:
        return str(value)
    return default


def build_profile_terminal_scope(hermes_home: "Any") -> Dict[str, str]:
    """Build the COMPLETE effective ``TERMINAL_*`` policy for a profile home.

    Projection order: defined defaults (``DEFAULT_CONFIG['terminal']``) ← the
    profile's ``.env`` TERMINAL_* selections ← its ``config.yaml`` explicit
    ``terminal:`` keys. The result is total: every key the terminal stack can
    ask for resolves from this mapping, so a bound scope never widens back to
    ambient process authority. Raises :class:`TerminalPolicyUnavailable` when
    either file exists but cannot be read/parsed (fail closed).
    """
    home = Path(hermes_home)

    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG.get("terminal") if isinstance(
        DEFAULT_CONFIG, dict) else None
    defaults = dict(defaults) if isinstance(defaults, dict) else {}
    # Terminal keys whose env mirror exists but whose config default lives in
    # the consuming tool rather than DEFAULT_CONFIG. These are the documented
    # tool-level defaults (tools/terminal_tool.py); without them the
    # projection would not be total and reads could observe nothing (which is
    # correct) OR fall back ambiently (which is not).
    defaults.setdefault("cwd", ".")           # per-surface placeholder
    defaults.setdefault("ssh_host", "")       # remote backends: unset = none
    defaults.setdefault("ssh_user", "")
    defaults.setdefault("ssh_port", 22)
    defaults.setdefault("ssh_key", "")
    defaults.setdefault("docker_orphan_reaper", True)
    defaults.setdefault("docker_persist_across_processes", True)
    defaults.setdefault("sandbox_dir", "")    # tool derives HERMES_HOME path
    defaults.setdefault("lifetime_seconds", 300)
    defaults.setdefault("docker_shared_container_key", "")
    defaults.setdefault("home_mode", "auto")

    scope: Dict[str, str] = {}

    def _apply(cfg_key: str, value: Any) -> None:
        if value is None:
            return
        # cwd placeholders (".", "auto", "cwd") are resolved per-surface
        # later; they are not a policy value.
        if cfg_key == "cwd" and str(value).strip() in {".", "auto", "cwd"}:
            return
        from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

        env_var = TERMINAL_CONFIG_ENV_MAP.get(cfg_key)
        if env_var:
            scope[env_var] = str(value)

    # 1) Defined defaults — the total baseline.
    for cfg_key, value in defaults.items():
        _apply(cfg_key, value)

    # 2) The profile's .env TERMINAL_* selections. Fail closed on unreadable
    #    files (missing file = no selections, fine).
    env_path = home / ".env"
    if env_path.exists():
        # Pre-flight readability: load_env_file swallows OSError/UnicodeError
        # by design (secret scope fails soft), but an unreadable profile .env
        # is a policy-resolution failure here and must fail closed.
        try:
            env_path.read_bytes()
        except Exception as exc:
            raise TerminalPolicyUnavailable(
                f"cannot read {env_path}: {exc}"
            ) from exc
        from agent.secret_scope import load_env_file

        selections = load_env_file(env_path)
        for key, value in selections.items():
            if key.startswith("TERMINAL_"):
                scope[key] = str(value)

    # 3) The profile's config.yaml explicit terminal keys. Read through the
    #    HERMES_HOME override so the profile's own file is consulted; a
    #    present-but-unparseable file fails closed (matches the gateway's
    #    _warn_config_parse_failure posture of refusing to guess policy).
    from hermes_constants import (
        get_hermes_home_override,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    override_token = None
    if get_hermes_home_override() != str(home):
        override_token = set_hermes_home_override(home)
    try:
        config_path = home / "config.yaml"
        if config_path.exists():
            # Parse the profile's file directly rather than through
            # read_raw_config(): that helper collapses "missing" and
            # "unparseable" into the same {} result. Here the file's existence
            # is already established, so {} can only mean a parse failure —
            # which must fail closed rather than silently projecting defaults.
            from hermes_cli.config import fast_safe_load

            try:
                with open(config_path, encoding="utf-8") as f:
                    raw = fast_safe_load(f)
            except Exception as exc:
                raise TerminalPolicyUnavailable(
                    f"cannot parse {config_path}: {exc}"
                ) from exc
            raw_terminal = raw.get("terminal") if isinstance(raw, dict) else None
            if isinstance(raw_terminal, dict):
                # The profile's file is parsed raw here, so any ${VAR} /
                # ${env:VAR} reference in terminal: keys is still literal.
                # Expand with the same resolver the startup config path uses:
                # _env_ref_lookup honors the active profile's secret scope
                # (installed before this per-turn scope), so a multiplexed
                # turn resolves against THIS profile's .env (#101659).
                from hermes_cli.config import _expand_env_vars

                for cfg_key, value in _expand_env_vars(raw_terminal).items():
                    _apply(cfg_key, value)
    except TerminalPolicyUnavailable:
        raise
    except Exception as exc:
        raise TerminalPolicyUnavailable(
            f"cannot resolve terminal config in {home}: {exc}"
        ) from exc
    finally:
        if override_token is not None:
            reset_hermes_home_override(override_token)

    return scope


def install_profile_terminal_scope(hermes_home: "Any") -> Token:
    """Build AND install a profile's policy in one call.

    The single entry point for every profile boundary (gateway turn, TUI/
    dashboard turn, cron fire). On resolution failure this installs the
    refusal scope instead of raising — the turn continues only in the sense
    that terminal tools will refuse execution with the typed reason; it never
    falls back to ambient process policy.

    Returns the token for ``reset_terminal_scope``.
    """
    try:
        return set_terminal_scope(build_profile_terminal_scope(hermes_home))
    except TerminalPolicyUnavailable as exc:
        logger.warning("terminal policy unavailable: %s", exc)
        return install_refusal_scope(str(exc))


def enforce_no_refusal() -> None:
    """Raise when the active scope is a refusal scope (fail closed).

    Execution paths (terminal tool, execute_code) call this before spawning
    anything: under a refusal scope the profile's terminal policy could not be
    resolved, and running with the launch process's ambient policy is exactly
    the authority leak this module closes (#68559 requires refusal, not
    fallback). Non-scoped and policy-scoped contexts pass silently.
    """
    scope = _terminal_scope_var.get()
    if isinstance(scope, TerminalPolicyRefusal):
        raise TerminalPolicyUnavailable(
            f"terminal policy unavailable for this profile: {scope.reason}"
        )


@contextmanager
def install_and_reset_profile_terminal_scope(
    hermes_home: "Any",
) -> Iterator[None]:
    """Install the profile's terminal policy for a bounded turn/fire.

    Single call for every in-process profile boundary (gateway turn,
    dashboard/TUI turn, cron fire): builds the complete effective policy and
    resets it on exit. Resolution failure installs the refusal scope for the
    same duration — terminal execution inside the block raises (fail closed)
    instead of inheriting the launch process's ambient policy. Never raises.
    """
    token = install_profile_terminal_scope(hermes_home)
    try:
        yield
    finally:
        reset_terminal_scope(token)
