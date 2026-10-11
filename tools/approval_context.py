"""Approval context: who is asking, from where, under which policy.

Session identity and observability contextvars, the interactive/gateway/cron/
unattended predicates, and the ``approvals.*`` config readers used by every
gate in :mod:`tools.approval`.
"""

import contextvars
import logging
import os
from agent.i18n import t
from hermes_cli.config import cfg_get
from utils import env_var_enabled, is_truthy_value

logger = logging.getLogger("tools.approval")


def _ctx(name: str, default: str | None = "") -> contextvars.ContextVar:
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


# Platforms where a *registered* gateway notify callback still does not mean a human can answer:
# the generic TurnRunner lane registers one for every inbound turn
# (``_run_conversation_with_approval`` registers unconditionally, no platform branch), while the
# adapter renders no ``send_exec_approval``/``/approve`` surface and the inbound lane is a
# fire-and-forget ``POST -> 202`` with no reader. Notifier presence is only a meaningful
# "someone can answer" discriminator on api_server, whose turn paths choose whether to register one.
_NOTIFIER_BLIND_APPROVAL_PLATFORMS = frozenset({"webhook", "msgraph_webhook"})


def _is_single_query_approval_context() -> bool:
    """True for a single-query (-q) session: ``hermes chat -q`` exports
    ``HERMES_INTERACTIVE=1`` (so sudo password prompts work) but nobody is waiting
    to answer approvals; without this marker the gate would wait the full timeout,
    fail closed and push the agent toward workarounds (e.g. execute_code).
    ``approvals.single_query_mode`` makes the path deterministic."""
    return is_truthy_value(_session_env("HERMES_SINGLE_QUERY_SESSION"))


def _no_user_can_answer() -> bool:
    """True in single-query (-q), cron and unattended-platform sessions. `hermes chat -q` still registers the
    CLI panel callback, so a prompt that only checks for a callback would wait the full timeout for nobody."""
    return (_is_single_query_approval_context() or _is_cron_approval_context()
            or _is_unattended_platform_approval_context())


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


def _get_smart_failure_threshold() -> int:
    """``approvals.smart_failure_threshold``: default 3; 0 or negative disables degradation.

    Counts consecutive times the guardian LLM could not be reached -- NOT consecutive
    DENYs, which ``approvals.denial_breaker_threshold`` already covers. Distinct knobs
    because they call for opposite responses: a DENY streak is the reviewer working,
    a FAILURE streak is the reviewer absent.

    Catches broad ``Exception``, not just ``ValueError``/``TypeError``: this is read from
    the guardian's *failure* path, and a config that cannot be read must yield the default
    rather than raise out of the code that is trying to record a failure. A guard that
    breaks when it is needed is worse than no guard.
    """
    try:
        return int(_get_approval_config().get("smart_failure_threshold", 3))
    except Exception:
        return 3


def _effective_approval_mode(session_key: str = "") -> str:
    """The mode to actually use, after degrading a failing guardian to manual.

    WHY THIS EXISTS. In ``smart`` mode a flagged command is graded by an auxiliary LLM.
    When that call fails -- 429, timeout, provider down, empty body -- the verdict is
    'escalate', which means "ask the human". If the human is not watching the surface
    that carries the prompt, the request sits until ``approvals.timeout`` and then fails
    closed: the agent is blocked for minutes and the user never saw a question. A silent
    five-minute stall is strictly worse than not asking, so after a run of failures we
    stop asking the guardian and go straight to manual.

    Degradation is per-session and self-healing: any real guardian answer clears the
    tally, so the moment the provider recovers, smart mode comes back on its own. It is
    deliberately one-way within a failure run -- there is no probing while failing, since
    every probe is a flagged command delayed by a doomed LLM call.

    Never raises: it is consulted on the gate path, and a broken config must not turn an
    approval question into a crash. If the config cannot be read at all it fails toward
    "manual", which is the same default `_get_approval_mode` uses.
    """
    try:
        mode = _get_approval_mode()
        if mode != "smart":
            return mode
        threshold = _get_smart_failure_threshold()
        if threshold <= 0 or not session_key:
            return mode
        from tools.approval_smart import _smart_failure_count
        if _smart_failure_count(session_key) >= threshold:
            return "manual"
    except Exception as e:
        logger.debug("Smart-approval degradation check failed, falling back to manual: %s", e)
        return "manual"
    return mode


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
        safe_cap = 300  # dependency failure must keep the safe default
    if raw > safe_cap:
        logger.warning("approvals.timeout=%s exceeds the platform-safe maximum; clamping to %ss", raw, safe_cap)
    return min(raw, safe_cap)


def format_approval_window(seconds: int) -> str:
    """The ONE human wording for an approval timeout window, shared by the CLI timeout notice,
    the tool result's ``user_summary`` and the gateway card copy so every surface agrees:
    300 → "5 minutes", 90 → "90 seconds", 7200 → "2 hours"."""
    seconds = max(int(seconds or 0), 0)
    if seconds and seconds % 3600 == 0:
        count, unit = seconds // 3600, "hour"
    elif seconds and seconds % 60 == 0:
        count, unit = seconds // 60, "minute"
    else:
        count, unit = seconds, "second"
    return t(f"approval.window.{unit}_one" if count == 1 else f"approval.window.{unit}_other", count=count)


def approval_timeout_notice_kwargs() -> dict:
    """``{waited, suggested}`` for the ``approval.timeout`` copy: how long we waited (``5 minutes`` /
    ``90 seconds``) and a tripled ``approvals.timeout`` value the user can paste into ``hermes config set``."""
    seconds = _get_approval_timeout()
    return {"waited": format_approval_window(seconds), "suggested": seconds * 3}


# --- Smart-approval failure tally, re-exported for tools/approval_smart ---------------------------------
# The tally lives in approval_smart (beside the call that produces the failures); these
# re-exports let that module report without importing approval.py, which imports it. Same
# direction as _fire_approval_hook above, for the same reason: a cycle here is a real one.

def mark_smart_failure() -> None:
    """Record that the guardian LLM could not be reached for the current session.

    Never raises, and never lets a config problem escape: it is called from the guardian's
    failure path, so an exception here would convert a recoverable "reviewer unavailable"
    into a broken approval gate.
    """
    try:
        from tools.approval_smart import _record_smart_failure
        count = _record_smart_failure(get_current_session_key())
        threshold = _get_smart_failure_threshold()
        if threshold > 0 and count == threshold:
            logger.warning(
                "Smart approvals: guardian unreachable %d times in a row (threshold %d) — "
                "degrading this session to manual approvals until it answers again",
                count, threshold,
            )
    except Exception as e:
        logger.debug("Smart-approval failure recording failed (ignored): %s", e)


def mark_smart_success() -> None:
    """Record that the guardian answered, clearing any failure run. Never raises."""
    try:
        from tools.approval_smart import _reset_smart_failures
        _reset_smart_failures(get_current_session_key())
    except Exception as e:
        logger.debug("Smart-approval success recording failed (ignored): %s", e)


def smart_approval_failure_notice(session_key: str = "") -> str:
    """'' unless the guardian has failed enough to have degraded this session, else a notice.

    This is the surfacing half of the degradation. The point of degrading is that the user
    stops getting prompts they cannot see; that is only an improvement if they are TOLD the
    security reviewer stopped working, because the alternative reading of the same event is
    "the reviewer is fine and this command was approved". Silence would be a security-
    relevant lie, so the notice is explicit that nothing was assessed.

    Never raises: it is appended to approval messages, and a broken config must not turn a
    denial into a crash.
    """
    try:
        from tools.approval_smart import _smart_failure_count
        threshold = _get_smart_failure_threshold()
        key = session_key or get_current_session_key()
        count = _smart_failure_count(key)
        if threshold <= 0 or count < threshold:
            return ""
    except Exception:
        return ""
    return (
        f"\n\nNOTE: the smart-approval security reviewer has been UNREACHABLE for the last "
        f"{count} flagged commands (its model is rate-limited or down), so no automated "
        f"assessment has run. This session has degraded to MANUAL approvals: every flagged "
        f"command is now being asked of you directly. Nothing here was auto-approved. "
        f"To restore automatic review, give the grader a fallback with "
        f"`hermes config set fallback_providers '[...]'`, or set `approvals.mode` to "
        f"`manual` to stop asking the reviewer at all."
    )


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
