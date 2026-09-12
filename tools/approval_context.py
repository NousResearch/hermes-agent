"""Approval context: who is asking, from where, under which policy.

Session identity and observability contextvars, the interactive/gateway/cron/
unattended predicates, and the ``approvals.*`` config readers used by every
gate in :mod:`tools.approval`.
"""

import contextvars
import logging
import os
from hermes_cli.config import cfg_get
from utils import env_var_enabled, is_truthy_value

logger = logging.getLogger("tools.approval")


def _ctx(name: str, default: "str | None" = "") -> contextvars.ContextVar:
    return contextvars.ContextVar(name, default=default)


# Per-thread/per-task gateway session identity: gateway runs agent turns concurrently in executor threads, so a
# process-global env var is racy (the env fallback stays for legacy single-threaded callers).
_approval_session_key: contextvars.ContextVar[str] = _ctx("approval_session_key")
_approval_turn_id: contextvars.ContextVar[str] = _ctx("approval_turn_id")
_approval_tool_call_id: contextvars.ContextVar[str] = _ctx("approval_tool_call_id")
# Hermes session id (observability identity, distinct from the gateway routing session_key), forwarded to approval
# hooks so observer plugins attach marks to the REAL session scope — otherwise they fall back to a synthetic "default"
# session whose scope never closes, so close-time exporters never ship them.
_approval_session_id: contextvars.ContextVar[str] = _ctx("approval_session_id")
# Interactive-CLI flag. Concurrent ACP sessions share a ThreadPoolExecutor, so mutating
# os.environ["HERMES_INTERACTIVE"] races: one session's `finally` restore can clobber another's set mid-run, dropping
# it onto the non-interactive auto-approve path so a dangerous command runs without the approval callback firing
# (GHSA-96vc-wcxf-jjff). None = unset → env fallback.
_hermes_interactive_ctx: contextvars.ContextVar[str | None] = _ctx("hermes_interactive", None)


def set_hermes_interactive_context(interactive: bool) -> contextvars.Token:
    """Bind interactive mode for the current context instead of mutating os.environ."""
    return _hermes_interactive_ctx.set("1" if interactive else "")


def reset_hermes_interactive_context(token: contextvars.Token) -> None:
    """Restore the prior value from :func:`set_hermes_interactive_context`."""
    _hermes_interactive_ctx.reset(token)


def _is_interactive_cli() -> bool:
    """True for an interactive CLI/ACP session (contextvar first, env fallback)."""
    ctx_val = _hermes_interactive_ctx.get()
    return is_truthy_value(ctx_val) if ctx_val is not None else env_var_enabled("HERMES_INTERACTIVE")


def _fire_approval_hook(hook_name: str, **kwargs) -> None:
    """Invoke a plugin lifecycle hook (pre_approval_request / post_approval_response).

    Lazy-imports the plugin manager (approval.py is imported long before plugins
    are discovered). Never raises: approval flow is safety-critical, plugin
    observability is not.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook
    except Exception:
        return  # plugin system unavailable (bare tool-only imports, minimal tests)
    try:
        kwargs.setdefault("turn_id", _approval_turn_id.get())
        kwargs.setdefault("tool_call_id", _approval_tool_call_id.get())
        if _approval_session_id.get():
            kwargs.setdefault("session_id", _approval_session_id.get())
        invoke_hook(hook_name, **kwargs)
    except Exception as exc:
        # invoke_hook() swallows per-callback errors; this is the dispatch layer itself failing.
        logger.debug("Approval hook %s dispatch failed: %s", hook_name, exc)


def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


_Tokens = tuple[contextvars.Token[str], contextvars.Token[str], contextvars.Token[str]]


def set_current_observability_context(*, turn_id: str = "", tool_call_id: str = "", session_id: str = "") -> _Tokens:
    """Bind active tool correlation IDs to approval hooks."""
    return (_approval_turn_id.set(turn_id or ""), _approval_tool_call_id.set(tool_call_id or ""),
            _approval_session_id.set(session_id or ""))


def reset_current_observability_context(tokens: _Tokens) -> None:
    """Restore prior approval hook correlation IDs."""
    turn_token, tool_token, session_token = tokens
    _approval_session_id.reset(session_token)
    _approval_tool_call_id.reset(tool_token)
    _approval_turn_id.reset(turn_token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key: approval contextvar → session_context → os.environ."""
    if session_key := _approval_session_key.get():
        return session_key
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", default)


def _session_env(name: str) -> str:
    """Session-scoped env value, contextvar-first so one cron/-q job cannot taint
    unrelated gateway/API/TUI turns in the same process; process env is the
    fallback for CLI tests and older entrypoints."""
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, "") or ""
    except Exception:
        return os.getenv(name, "") or ""


def _get_session_platform() -> str:
    """Return the current gateway platform from contextvars/env fallback."""
    return _session_env("HERMES_SESSION_PLATFORM")


def _is_cron_approval_context() -> bool:
    """True when the current approval decision is running inside cron."""
    return is_truthy_value(_session_env("HERMES_CRON_SESSION"))


# Programmatic/unattended platforms: no human can answer a prompt and the adapter has no ``send_exec_approval`` /
# ``/approve`` surface. Governed by ``approvals.unattended_mode`` (default deny), mirroring ``cron_mode`` — never an
# interactive round-trip that blocks for the full timeout with nobody to answer.
_UNATTENDED_APPROVAL_PLATFORMS = frozenset({"webhook", "msgraph_webhook", "api_server"})


def _is_unattended_platform_approval_context() -> bool:
    """True when the session platform is a programmatic/unattended surface.

    Webhook, msgraph_webhook, and api_server sessions bind ``HERMES_SESSION_PLATFORM`` like chat gateways
    do, but there is no human who can resolve a pending approval. Treating them as gateway approval contexts
    blocks the session for the full approval timeout (60-300s) and then fails closed anyway — the deadlock
    in #37284/#87509.
    """
    return _get_session_platform() in _UNATTENDED_APPROVAL_PLATFORMS


def _is_single_query_approval_context() -> bool:
    """True for a single-query (-q) session: ``hermes chat -q`` exports
    ``HERMES_INTERACTIVE=1`` (so sudo password prompts work) but nobody is waiting
    to answer approvals; without this marker the gate would wait the full timeout,
    fail closed and push the agent toward workarounds (e.g. execute_code).
    ``approvals.single_query_mode`` makes the path deterministic."""
    return is_truthy_value(_session_env("HERMES_SINGLE_QUERY_SESSION"))


def _is_gateway_approval_context() -> bool:
    """True inside a gateway/API session that can answer an approval.

    Legacy integrations set HERMES_GATEWAY_SESSION; concurrent paths bind
    HERMES_SESSION_PLATFORM via contextvars. Cron is NEVER a gateway approval
    context even when it originated from a platform (cron binds the platform for
    delivery routing): falling through would submit a pending approval with no
    listener and block the job indefinitely; unattended platforms likewise.

    Unattended programmatic platforms (webhook, msgraph_webhook, api_server) are excluded for the same
    reason: those adapters have no ``send_exec_approval`` and no way to receive ``/approve`` replies.
    Submitting a pending approval there blocks the session for the full approval timeout (60-300 s) with no
    human who can resolve it (#37284, 87509). Their dangerous-command handling is governed by
    ``approvals.unattended_mode`` config (default deny), mirroring cron.
    """
    if _is_cron_approval_context() or _is_unattended_platform_approval_context():
        return False
    return env_var_enabled("HERMES_GATEWAY_SESSION") or bool(_get_session_platform())


def _resolve_cli_approval_callback(approval_callback=None):
    """Explicit callback, else the per-thread one from ``terminal_tool.set_approval_callback``."""
    if approval_callback is not None:
        return approval_callback
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback()
    except Exception:
        return None


def _should_fall_through_to_cli_approval(*, is_cli: bool, approval_callback, notify_cb) -> bool:
    """Prefer the CLI Dangerous Command panel over a silent pending approval:
    ``HERMES_EXEC_ASK`` (or a platform marker) can leak into an interactive CLI
    process (historically via ``import gateway.run``), and without a gateway notify
    listener the ask branch used to return ``pending_approval`` immediately and
    skip the panel the user can actually answer."""
    return bool(is_cli and approval_callback is not None and notify_cb is None)


_VALID_MODES = ("manual", "smart", "off")


def _normalize_approval_mode(mode) -> str:
    """Normalize approval mode values loaded from YAML/config. YAML 1.1 parses a
    bare ``off`` as False, so ``mode: off`` arrives as a bool; treat it as the
    intended string mode. Unknown strings (e.g. 'auto') warn and fall back to
    'manual' instead of silently failing every mode check."""
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in _VALID_MODES:
            return normalized
        if normalized:
            logger.warning("Unknown approvals.mode %r — defaulting to 'manual'. "
                           "Valid values: %s", mode, ", ".join(_VALID_MODES))
    return "manual"


def _get_approval_config() -> dict:
    """Read the approvals config block: the LIVE config-cache sub-dict
    (load_config_readonly contract) — callers must not mutate it or any nested structure."""
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly().get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """Return 'manual', 'smart', or 'off' (a hosted-room policy overrides config)."""
    try:
        from gateway.hosted_room_execution_policy import current_room_execution_policy
        if (room_policy := current_room_execution_policy()) is not None:
            return room_policy.approval_mode
    except Exception:
        pass
    return _normalize_approval_mode(_get_approval_config().get("mode", "manual"))


def _get_approval_timeout() -> int:
    """Read ``approvals.timeout`` (default 300s: gateway push notifications may
    not be seen for minutes; 60s failed closed before Telegram taps landed).
    Clamped to ``agent.deadline.MAX_SAFE_TIMEOUT_S`` (~1 year): a larger value
    overflows ``time_t`` inside ``Thread.join`` / ``Lock.acquire`` on macOS and
    crashed every parallel tool batch; clamping at the single config-read site
    keeps every consumer platform-safe at once."""
    try:
        raw = int(_get_approval_config().get("timeout", 300))
    except (ValueError, TypeError):
        return 300
    try:
        from agent.deadline import MAX_SAFE_TIMEOUT_S
        safe_cap = int(MAX_SAFE_TIMEOUT_S)
    except Exception:
        safe_cap = 365 * 24 * 3600  # fail CLOSED: the raw value would re-open the overflow
    if raw > safe_cap:
        logger.warning("approvals.timeout=%s exceeds the platform-safe maximum; clamping to %ss", raw, safe_cap)
    return min(raw, safe_cap)


def _binary_approval_mode(key: str) -> str:
    """Read ``approvals.<key>`` as 'approve' or 'deny' (default deny)."""
    try:
        from hermes_cli.config import load_config_readonly
        mode = str(cfg_get(load_config_readonly(), "approvals", key, default="deny")).lower().strip()
        return "approve" if mode in {"approve", "off", "allow", "yes"} else "deny"
    except Exception:
        return "deny"


def _get_cron_approval_mode() -> str:
    """Read the cron approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("cron_mode")


def _get_single_query_approval_mode() -> str:
    """Read the single-query (-q) approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("single_query_mode")


def _get_unattended_approval_mode() -> str:
    """Approval mode for webhook / msgraph_webhook / api_server sessions; default
    deny — an unattended session never silently runs a flagged action unless the
    operator explicitly trusts it."""
    return _binary_approval_mode("unattended_mode")


def _tirith_fail_open() -> bool:
    """``security.tirith_fail_open`` (default True; True when config is unreadable).
    False means the operator opted into fail-closed: an un-importable scanner
    must not silently grant access."""
    try:
        from hermes_cli.config import load_config_readonly
        _sec = (load_config_readonly() or {}).get("security", {}) or {}
        return bool(_sec.get("tirith_fail_open", True)) if _sec.get("tirith_enabled", True) else True
    except Exception:
        return True


def _get_approval_transport_config() -> tuple[str, str | None]:
    """Return explicitly selected transport and fail-closed fallback mode."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = ((load_config_readonly() or {}).get("security") or {}).get("approval") or {}
        selected = str(cfg.get("transport") or "builtin").strip().lower()
        fallback = str(cfg.get("transport_fallback") or "").strip().lower()
    except Exception:
        # An unreadable/malformed selection must not silently materialize a
        # prompt on a built-in surface the operator may not be watching.
        return "config-error", None
    return selected or "builtin", "builtin" if fallback == "builtin" else None


# --- Operator-qualified security-policy write context (#81108, #104697 review) --------
# The ONLY mechanism that authorizes mutating a security-policy config key is a
# one-shot grant stamped by a HUMAN-ACTOR code path and CONSUMED by the writer:
#
#   grant_operator_policy_write(actor) → token → consume_operator_policy_write()
#
# Both ends are here, and the writer additionally requires a human-actor CONTEXT.
# Why this shape (and not a parameter, an env var, or a reusable boolean scope):
# the #104697 review established that any *reusable* importable token is forgeable
# by the agent process that can import the writer (#104059 class). A one-shot grant
# is forgeable too — agent-executed Python in-process CAN call the granter — so the
# writer's CONTEXT check is the load-bearing half: the granter is only invoked by
# the sanctioned human-actor paths, the grant is single-use (minted and consumed
# within one write), and the context check refuses in any headless agent context
# (cron / -q / unattended / detector-approved child processes). The gateway branch
# of the context check was REMOVED after round-2 review: a gateway backend process
# is exactly where agent turns run, so "platform env set" proves nothing about a
# human — the gateway /approvals path instead runs the writer with a grant stamped
# AFTER its enabled-admin-policy check (the human is the authenticated sender of
# the slash command), and the grant is consumed before any agent code can race it.
_operator_policy_write_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "operator_policy_write", default=False)


def grant_operator_policy_write() -> "contextvars.Token[bool]":
    """Stamp a one-shot operator-policy write grant. Call ONLY from a sanctioned
    human-actor path: the gateway /approvals handler (after its enabled-admin
    check — the human is the authenticated command sender) or the interactive
    operator CLI (the human typing the command)."""
    return _operator_policy_write_ctx.set(True)


def reset_operator_policy_write(token: "contextvars.Token[bool]") -> None:
    _operator_policy_write_ctx.reset(token)


def is_operator_policy_write() -> bool:
    """True inside an operator-qualified write scope (see module comment)."""
    return _operator_policy_write_ctx.get()


def _is_human_actor_context() -> bool:
    """True when THIS process is driven by a human as the actor: an interactive
    operator CLI with a TTY on stdin. Deliberately does NOT trust gateway
    platform env: a gateway backend process is where agent turns execute, so
    "platform env set" carries no human proof (round-2 review finding). The
    gateway /approvals path authorizes via the one-shot grant stamped after its
    admin check, not via this context test."""
    if _is_interactive_cli():
        # A real operator CLI has a TTY on stdin; a scripted/headless child does not.
        try:
            import sys
            if sys.stdin is not None and sys.stdin.isatty():
                return True
        except Exception:
            pass
    return False


def _get_trusted_execute_code_profiles() -> list[str]:
    """Read ``approvals.trusted_execute_code_profiles``: an explicit allowlist of
    profile names trusted for whole-script execute_code auto-approval.

    Returns a list (never a non-list); a malformed value is treated as empty (safe).
    Reuses ``_get_approval_config()`` (same module) instead of re-loading the config.
    """
    try:
        raw = _get_approval_config().get("trusted_execute_code_profiles", []) or []
        if not isinstance(raw, (list, tuple, set)):
            return []
        return [str(item).strip() for item in raw if item is not None and str(item).strip()]
    except Exception:
        return []


def _execute_code_profile_is_trusted() -> bool:
    """True when the ACTIVE Hermes profile is explicitly trusted for whole-script
    execute_code auto-approval. Profile-scoped via ``get_active_profile_name``
    (inferred from HERMES_HOME); the config block is itself profile-specific, so
    trust never leaks across profiles. Lazy-import avoids any import cycle with
    hermes_cli.profiles (matching run_agent.py's pattern).

    Fail-safe: any error resolving the profile (import failure, HERMES_HOME
    resolution error, a raising ``get_active_profile_name``) yields ``False`` — an
    unexpected failure must never accidentally grant trust."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = (get_active_profile_name() or "").strip().lower()
    except Exception:
        logger.debug("execute_code trust lane disabled: profile resolution failed", exc_info=True)
        return False
    trusted = [p.lower() for p in _get_trusted_execute_code_profiles()]
    return bool(profile) and profile in trusted


def trusted_execute_code_status_line() -> str:
    """/status line: the active profile's execute_code trust state (#44993)."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name()
    except Exception:
        profile = None

    if not profile:
        return "execute_code trust: (no active profile)"

    try:
        is_trusted = _execute_code_profile_is_trusted()
    except Exception:
        is_trusted = False

    status_str = "trusted" if is_trusted else "not trusted"
    return f"execute_code trust: {profile} ({status_str})"
