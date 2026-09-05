"""Approval bridge for external agent-loop runtimes (claude-agent-sdk) under the gateway.

The SDK's per-tool permission request is bridged onto the same per-session queue + notify
machinery the native terminal guard uses (:func:`tools.approval_gateway_wait._await_gateway_decision`),
so the user gets a real chat approval prompt instead of a silent SDK-internal deny. The
callback returns a plain choice string ("once"/"deny") for the classic outcomes, or a dict
``{"choice": str, "reason": str}`` when the deny needs an HONEST reason: "denied by user" is
reserved for a human actually choosing deny — timeout, no-approver and notify-failure each
carry their own reason (the model used to hear "denied by user" for prompts no user ever saw —
2026-08-06 incident). (#25267)
"""

from __future__ import annotations

import logging
import weakref

from hermes_cli.lifecycle import observer_failure_log
from tools import approval as _approval
from tools import approval_context as _ctx
from tools import approval_gateway_wait, approval_session_notify, approval_smart
from utils import env_var_enabled

logger = logging.getLogger("tools.approval")

_SDK_PATTERN_KEY = "claude_sdk_tool"
_MAX_SDK_APPROVAL_CONTEXT_FIELDS = 8


def current_approval_turn_context() -> dict:
    """Snapshot the TURN-variant approval context for later use on a thread where contextvars
    are invisible. Runtimes whose approval callback fires on a foreign thread (the claude-agent-sdk
    loop) call this at the top of EVERY turn — on the agent turn thread, where the session
    contextvars are visible — and hand the snapshot to :func:`build_sdk_gateway_approval_callback`
    via its ``context_provider``."""
    return {"gateway": _ctx._is_gateway_approval_context(), "session_key": _ctx.get_current_session_key("")}


def _normalize_sdk_tool_use_id(value: object) -> str:
    """Keep opted-in SDK correlation metadata exact, meaningful, and bounded (≤256 UTF-8 bytes,
    printable, no whitespace, no surrogates); anything else correlates nothing."""
    if type(value) is not str or not value:
        return ""
    utf8_bytes = 0
    for char in value:
        code_point = ord(char)
        if code_point <= 0x7F:
            utf8_bytes += 1
        elif code_point <= 0x7FF:
            utf8_bytes += 2
        elif 0xD800 <= code_point <= 0xDFFF:
            return ""
        elif code_point <= 0xFFFF:
            utf8_bytes += 3
        else:
            utf8_bytes += 4
        if utf8_bytes > 256:
            return ""
        if not char.isprintable() or char.isspace():
            return ""
    return value


def sdk_bash_immutable_floor_reason(command: object) -> str | None:
    """Return a side-effect-free immutable SDK Bash denial, if any.

    The single policy owner shared by the mandatory SDK session wrapper: no warning, tally,
    observer, queue, or card side effects."""
    if type(command) is not str:
        return "canonical request is unassessable"
    hardline, hardline_description = _approval.detect_hardline_command(command)
    if hardline:
        return (f"BLOCKED (hardline): {hardline_description}. This SDK request is on the "
                "unconditional blocklist and cannot be executed via the agent.")
    sudo_guess, sudo_description = _approval._check_sudo_stdin_guard(command)
    if sudo_guess:
        return _approval._sudo_stdin_block_result(sudo_description)["message"]
    denied_by = _approval._match_user_deny_rule(command)
    if denied_by is not None:
        return _approval._user_deny_block_result(denied_by)["message"]
    return None


# --- Trust registry: exact object identity of callbacks built here (no equality/hash keyed fallback) ---
_trusted_sdk_gateway_approval_callbacks: list[weakref.ReferenceType] = []


def _live_trusted_refs(exclude=None) -> list[weakref.ReferenceType]:
    return [ref for ref in _trusted_sdk_gateway_approval_callbacks
            if ref is not exclude and ref() is not None]


def _discard_dead_trusted_sdk_gateway_approval_callback(dead_ref: weakref.ReferenceType) -> None:
    """Remove a finalized callback without invoking referent equality or hashing."""
    with _approval._lock:
        _trusted_sdk_gateway_approval_callbacks[:] = _live_trusted_refs(exclude=dead_ref)


def _register_trusted_sdk_gateway_approval_callback(callback: object) -> bool:
    """Register a weakly representable callable by exact object identity."""
    if callback is None or not callable(callback):
        return False
    try:
        callback_ref = weakref.ref(callback, _discard_dead_trusted_sdk_gateway_approval_callback)
    except TypeError:
        # Trust must not require a strong reference or an equality/hash keyed fallback. The
        # production gateway closure is weak-referenceable.
        return False
    with _approval._lock:
        live_refs = _live_trusted_refs()
        if not any(ref() is callback for ref in live_refs):
            live_refs.append(callback_ref)
        _trusted_sdk_gateway_approval_callbacks[:] = live_refs
    return True


def is_trusted_sdk_gateway_approval_callback(callback: object) -> bool:
    """Return whether the exact callback object was built by the SDK bridge."""
    if callback is None or not callable(callback):
        return False
    with _approval._lock:
        live_refs = _live_trusted_refs()
        _trusted_sdk_gateway_approval_callbacks[:] = live_refs
        return any(ref() is callback for ref in live_refs)


def _resolve_sdk_approval_context(context_provider, build_key: str, build_gateway: bool) -> tuple[str, bool]:
    """Per-call ``(session_key, gateway)`` resolution (P1.b — sticky-session binding fix).

    The cron veto and the session key change from turn to turn on one long-lived SDK session;
    freezing them at creation made a session first created during a CRON turn silently deny every
    un-allowlisted tool FOREVER, even in later interactive turns. Order: (1) live contextvar read —
    wins on a thread that can see the turn's context; (2) ``context_provider()`` — the runtime's
    per-turn snapshot, the production path (the SDK invokes the callback from its own loop thread,
    where session contextvars are NEVER propagated); (3) build-time snapshot for provider-less
    callers. The discriminator is the LIVE KEY, not the live gateway check: on a TUI SDK-thread call
    the process-level HERMES_GATEWAY_SESSION env is visible while the key contextvar is not."""
    live_key = _ctx.get_current_session_key("")
    if live_key:
        return live_key, _ctx._is_gateway_approval_context()
    if context_provider is None:
        return build_key, build_gateway
    try:
        ctx = context_provider()
        if (type(ctx) is not dict or len(ctx) > _MAX_SDK_APPROVAL_CONTEXT_FIELDS
                or "gateway" not in ctx or "session_key" not in ctx):
            raise TypeError("malformed SDK approval context")
        session_key, gateway = ctx["session_key"], ctx["gateway"]
        if type(session_key) is not str or type(gateway) is not bool:
            raise TypeError("malformed SDK approval context fields")
        return session_key, gateway
    except Exception:
        logger.debug("SDK approval context_provider failed at protected boundary")
        return "", False


def _sdk_smart_verdict(canonical_tool_input: str, safe_command: str, safe_description: str,
                       session_key: str) -> str:
    """Guardian step for an SDK request: the evaluator sees the bounded canonical SDK bytes;
    observers see only the bounded canonical-derived presentation. Every failure logs a fixed
    message — never the payload."""
    with observer_failure_log(_ctx.SDK_PRE_OBSERVER_FAILURE_LOG):
        observer = approval_smart._prepare_smart_approval_observer(
            command=safe_command, description=safe_description, pattern_key=_SDK_PATTERN_KEY,
            pattern_keys=[_SDK_PATTERN_KEY], session_key=session_key,
        )
    with observer_failure_log(_ctx.SDK_GUARDIAN_FAILURE_LOG):
        verdict = approval_smart._smart_approve(canonical_tool_input, safe_description)
    with observer_failure_log(_ctx.SDK_POST_OBSERVER_FAILURE_LOG):
        approval_smart._observe_smart_approval_verdict(observer, verdict)
    return verdict


def _sdk_decision_outcome(decision: dict, session_key: str):
    """Map the gateway wait's decision onto the widened SDK return channel."""
    if decision.get("notify_failed"):
        return {"choice": "deny",
                "reason": "approval request could not be delivered to the operator (notify failed)"}
    if not decision.get("resolved"):
        # The operator saw the prompt (notify succeeded) but never answered within the timeout.
        return {"choice": "deny", "reason": "approval timed out — no operator response"}
    choice = decision.get("choice")
    if choice in (None, "expired"):
        # Turn teardown expired this prompt before anyone answered: the model must never hear
        # "denied by user" for a prompt that simply died with the turn. (None is defensive
        # residue: teardown now always stamps "expired".)
        return {"choice": "deny", "reason": "approval expired (turn ended)"}
    if choice == "deny":
        reason = decision.get("reason")
        # Human attribution is a structural field emitted only by this registered bridge;
        # callback-controlled reason prefixes are never trusted by the SDK session layer.
        return {"choice": "deny", "operator_denial": True, "reason": reason if type(reason) is str else ""}
    # Clamp durable choices to one-shot: an older client button can still send "session"/"always";
    # the grant must not outlive the single SDK permission request it answered.
    _approval._reset_denials(session_key)
    return "once"


def build_sdk_gateway_approval_callback(context_provider=None):
    """Build the SDK approval callback, or None for surfaces that are not gateway-shaped (no
    ``HERMES_GATEWAY_SESSION`` env and no session platform binding): the CLI keeps its
    thread-local callback (tools.terminal_tool) and one-shot processes keep the SDK's settings
    posture. Build-time invariance proof: ``HERMES_GATEWAY_SESSION`` is process-level (tui_gateway
    sets it once at startup; nothing unsets it), and a process where no code ever binds a session
    platform has no path that starts binding one mid-lifetime — so "not gateway-shaped at
    SDK-session creation" cannot flip for that session's lifetime.

    A cron turn resolves gateway=False and lands in the honest no-approver deny WITHOUT paging or
    blocking — cron's allowlist-or-deny posture is preserved because settings allow-rules suppress
    prompts before ``can_use_tool`` is ever consulted; only would-be prompts reach this callback.

    Known v1 semantics (deliberate, documented trade-offs): the wait runs on the SDK loop's
    ``asyncio.to_thread`` worker, not the agent execution thread — the wait loop's
    ``is_interrupted()`` fast-deny and activity heartbeats do not apply; /stop interrupts the TURN
    at the SDK level and the orphaned wait self-expires at the approval timeout. The wait spends
    the SDK turn's own budget (``run_turn`` ``turn_timeout``, 600s): one ignored prompt costs the
    approval timeout (default 300s); a second can time the turn out, which retires the session
    (digest-resume next turn) and orphans the displayed prompt. Operators raising
    ``approvals.timeout`` should keep it under the agent watchdog timeout.
    """
    gateway_shaped = env_var_enabled("HERMES_GATEWAY_SESSION") or bool(_ctx._get_session_platform())
    if not gateway_shaped:
        return None
    # Build-time snapshots: the fallback for provider-less callers only.
    build_gateway = _ctx._is_gateway_approval_context()
    build_key = _ctx.get_current_session_key("")

    def _sdk_gateway_approval(command: str, description: str, *, allow_permanent: bool = False,
                              tool_use_id: str = "", canonical_tool_input: str | None = None):
        # Validate the SDK-owned canonical serialization before context providers, logs, queue/card
        # state, or any approval extension. The positional presentation arguments remain ABI-only
        # and are untrusted.
        try:
            from agent.transports.claude_agent_sdk_session import (
                safe_sdk_tool_presentation_from_canonical, validate_canonical_sdk_request_serialization,
            )

            canonical_request = validate_canonical_sdk_request_serialization(canonical_tool_input)
            safe_presentation = safe_sdk_tool_presentation_from_canonical(
                canonical_tool_input, _validated_request=canonical_request)
        except Exception:
            canonical_request = safe_presentation = None
        if canonical_request is None or safe_presentation is None:
            return {"choice": "deny", "reason": "canonical request is unassessable"}
        canonical_tool_input, _request = canonical_request
        safe_command, safe_description = safe_presentation

        session_key, gateway = _resolve_sdk_approval_context(context_provider, build_key, build_gateway)
        notify_cb = None
        if gateway and session_key:
            # Turn registration wins; the session-scoped entry is the between-turns fallback that
            # lets a background SDK turn page the operator.
            notify_cb = (_approval._gateway_notify_cb(session_key)
                         or approval_session_notify.lookup_session_notify(session_key))
        if notify_cb is None:
            # No approver anywhere (background context with no session registration, or gateway
            # tearing down): fail closed, but say so honestly — never attribute this deny to the user.
            logger.warning("SDK approval request has no approver available; denying without user attribution")
            return {"choice": "deny", "reason": "no approver available (background context)"}

        # The Smart decision seam runs only after the canonical request and approver context validated.
        smart_denied = False
        if _ctx._get_approval_mode() == "smart":
            verdict = _sdk_smart_verdict(canonical_tool_input, safe_command, safe_description, session_key)
            if verdict == "approve":
                _approval._reset_denials(session_key)
                return "once"
            if verdict == "deny":
                _approval._record_denial(session_key)
                smart_denied = True
        approval_data = {
            "command": safe_command, "pattern_key": _SDK_PATTERN_KEY, "pattern_keys": [_SDK_PATTERN_KEY],
            "description": safe_description,
            # One-tap "once" grants only: durable grants for SDK tools belong in the operator's
            # settings.json (setting_sources), where they are auditable — not in chat-tap persistence.
            "allow_permanent": False, "allow_session": False,
            # P2.a correlator: rides onto the pending _ApprovalEntry (entry.data IS this dict) so a
            # button tap can resolve the MATCHING prompt instead of queue[0].
            "tool_use_id": _normalize_sdk_tool_use_id(tool_use_id),
            # SDK cards carry per-request correlation and one-shot consent; they must remain distinct
            # even when their bounded summaries are identical, so they opt out of coalescing.
            "no_coalesce": True,
        }
        if smart_denied:
            approval_data["smart_denied"] = True

        def _sdk_safe_notify(data):
            try:
                notify_cb(data)
            except Exception:
                # The shared waiter logs exception text for legacy surfaces; replace only this SDK
                # callback's hostile exception with a fixed message.
                raise RuntimeError("SDK approval notification failed") from None

        decision = approval_gateway_wait._await_gateway_decision(session_key, _sdk_safe_notify, approval_data, surface="claude_sdk")
        return _sdk_decision_outcome(decision, session_key)

    # Markers for the SDK session layer: this callback accepts the correlator and canonical input
    # kwargs (the CLI thread-local callback does not — the session only passes them to callbacks
    # that opt in).
    _sdk_gateway_approval._accepts_tool_use_id = True
    _sdk_gateway_approval._accepts_canonical_tool_input = True
    if not _register_trusted_sdk_gateway_approval_callback(_sdk_gateway_approval):
        # A builder-owned Python closure is weak-referenceable; keep an unexpected runtime that
        # cannot represent identity out of trust.
        logger.warning("SDK gateway callback could not be registered as trusted")
    return _sdk_gateway_approval
