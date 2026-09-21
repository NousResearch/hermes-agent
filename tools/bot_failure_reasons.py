"""Typed failure-reason codes for bot turns and relay replies.

A closed vocabulary of machine-readable reason codes carried ALONGSIDE the
free-text ``error`` fields (additive — old consumers keep working). Platform-side
codes are assigned by the transport/relay layer; agent-side codes are derived
from raw agent/provider error text via ``classify_agent_error``. Classifier
precedence is the order of ``_RULES``: auth outranks quota by design — real
provider 401 bodies (e.g. Anthropic) say "invalid, blocked or out of funds".
"""

from __future__ import annotations

import re
from typing import Any

# platform-side
RUNTIME_OFFLINE = "runtime_offline"
QUEUED_EXPIRED = "queued_expired"
DELIVERY_TIMEOUT = "delivery_timeout"
AGENT_BLOCKED = "agent_blocked"
CANCELLED = "cancelled"
# Two refusals that used to ship the SAME code ('target_busy'), because a waiter only saw the
# human prose. They are different conditions with different operator actions (#93091 follow-up):
# TARGET_BUSY: another delivery TURN holds this profile's cross-process turn lock — a live turn,
#   or a delivery stuck in one; retry shortly, and a hold past the whole turn budget is a WEDGED
#   holder (tools.bot_relay names its pid and age).
# TARGET_SESSION_LIVE: the target's chat has a LIVE OWNER mid-turn — an interactive surface, or a
#   delivery session that outlived its requester (the orphan this code names). Nothing is queued
#   behind a turn here; the message was refused at the session lease, so retrying is the only
#   move and a session alive far past a turn is the watchdog signal
#   (hermes_cli.active_sessions.long_running_delivery_sessions).
TARGET_BUSY = "target_busy"
TARGET_SESSION_LIVE = "target_session_live"

# agent-side
PROVIDER_AUTH_OR_ACCESS = "provider_auth_or_access"
PROVIDER_QUOTA_LIMIT = "provider_quota_limit"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
PROVIDER_SERVER_ERROR = "provider_server_error"
CONTEXT_OVERFLOW = "context_overflow"
MISSING_CONFIG = "missing_config"
MODEL_UNAVAILABLE = "model_unavailable"
UNKNOWN = "unknown"

ALL_REASONS = frozenset({
    RUNTIME_OFFLINE, QUEUED_EXPIRED, DELIVERY_TIMEOUT, AGENT_BLOCKED, CANCELLED,
    PROVIDER_AUTH_OR_ACCESS, PROVIDER_QUOTA_LIMIT, PROVIDER_RATE_LIMIT,
    PROVIDER_SERVER_ERROR, CONTEXT_OVERFLOW, MISSING_CONFIG, MODEL_UNAVAILABLE, UNKNOWN,
})

#: Reasons a supervisor may retry automatically without human intervention.
AUTO_RETRYABLE = frozenset({RUNTIME_OFFLINE, DELIVERY_TIMEOUT, PROVIDER_RATE_LIMIT, PROVIDER_SERVER_ERROR})

#: The delivery-refusal codes a lane may put in an exception's ``reason`` and forward verbatim.
#: Kept BESIDE ``ALL_REASONS`` rather than inside it: ``ALL_REASONS`` is the agent/provider
#: vocabulary that is total over the retry policy (``retry_action``) and is the accepted set for
#: hosted-room ``turn.failed`` event payloads (``gateway.hosted_room_discussion``), whereas these
#: name the TARGET's state at admission, not the turn's failure class.
DELIVERY_REFUSAL_REASONS = frozenset({TARGET_BUSY, TARGET_SESSION_LIVE})

#: Every code that may appear in a delivery refusal's ``reason`` field: the agent/provider
#: vocabulary plus the two target-state refusals that predate it. Consumers that validate a
#: *delivery* reason (rather than a turn-failure reason) accept this set.
DELIVERY_REASONS = ALL_REASONS | DELIVERY_REFUSAL_REASONS


def is_auto_retryable(reason: str) -> bool:
    return reason in AUTO_RETRYABLE


# Retry session policy: a retried bot turn NEVER mints a fresh session. Transient
# classes resume as-is; context_overflow runs context compression (the one
# sanctioned context mutation) on the same session first; everything else
# (auth/quota/config/model/unknown) is never auto-retried — it can't be fixed by
# a retry and only burns quota.
# See #93091.
RETRY_RESUME = "resume"
RETRY_COMPRESS_THEN_RESUME = "compress_then_resume"
RETRY_NONE = "none"


def retry_action(reason: str) -> str:
    """Map a failure reason to the bot-turn retry action (see policy above)."""
    if reason in AUTO_RETRYABLE:
        return RETRY_RESUME
    if reason == CONTEXT_OVERFLOW:
        return RETRY_COMPRESS_THEN_RESUME
    return RETRY_NONE


_STATUS = r"(?:error code:?\s*|status(?:\s*code)?:?\s*|http\s*)"

# Ordered (pattern, code) — first match wins.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), code)
    for pat, code in (
        (rf"authentication_error|invalid api key|{_STATUS}(?:401|403)\b", PROVIDER_AUTH_OR_ACCESS),
        (rf"{_STATUS}402\b|out of funds|quota|balance", PROVIDER_QUOTA_LIMIT),
        (rf"{_STATUS}429\b|rate.?limit", PROVIDER_RATE_LIMIT),
        # ``server[ _]?error`` / ``overloaded_error`` are the providers' own JSON spellings of the
        # agent's typed ``server_error`` / ``overloaded`` verdicts (FailoverReason), which the
        # in-process lanes classify from ``failure_reason`` rather than a status number.
        (rf"{_STATUS}5\d{{2}}\b|server[ _]?error|overloaded", PROVIDER_SERVER_ERROR),
        (r"context length|context_overflow|maximum context", CONTEXT_OVERFLOW),
        (r"no llm provider configured|missing config|no access token", MISSING_CONFIG),
        (r"model .*(not found|does not exist)|model_not_found", MODEL_UNAVAILABLE),
    )
)


def turn_failure_text(stdout: str | None, stderr: str | None) -> str:
    """The error text of a failed ``hermes … -Q`` delivery turn: both streams, in the order the CLI
    writes them. The provider prose is the turn's final_response and lands on STDOUT; stderr carries
    session bookkeeping (``session_id: …``) on every run, so ``stderr or stdout`` only ever saw the
    banner and every transient failure classified as ``unknown``.

    The in-process lanes hand over the same two pieces of prose as an agent result dict
    (``error`` + ``failure_reason``); ``result_retry_action`` joins them here too, so one classifier
    sees every lane's failure text."""
    return "\n".join(text.strip() for text in (stdout, stderr) if text and text.strip())


def result_retry_action(result: Any) -> str:
    """The retry action for a finished IN-PROCESS turn (an ``_run_agent`` result dict).

    The transport-agnostic twin of the child-process lanes' ``retry_action(classify_agent_error(
    turn_failure_text(stdout, stderr)))``: same policy, the failure text just arrives as result fields
    instead of two streams — ``error`` carries the raw provider summary (status codes included) and
    ``failure_reason`` the turn loop's own typed verdict, which is what names an overflow the copy only
    describes in prose. ``RETRY_NONE`` for anything that did not fail (and for a non-dict result), so a
    successful turn whose text happens to mention 429 is never re-run."""
    if not isinstance(result, dict) or not result.get("failed"):
        return RETRY_NONE
    return retry_action(classify_agent_error(
        turn_failure_text(result.get("error"), result.get("failure_reason"))))


def classify_agent_error(text: str) -> str:
    """Map raw agent/provider error text to a closed reason code (``unknown`` when unmatched/empty)."""
    raw = str(text or "")
    if raw.strip():
        for pattern, code in _RULES:
            if pattern.search(raw):
                return code
    return UNKNOWN


# ``hermes_cli.active_sessions.SESSION_NOT_OWNED``, mirrored as a literal: this module is imported
# by both delivery lanes and must not drag in the CLI's session layer (import cycle). The two are
# pinned together by ``tests/tools/test_bot_failure_reasons.py``.
SESSION_NOT_OWNED_REASON = "SESSION_NOT_OWNED"
_REFUSAL_MARKER = "hermes-refusal-reason: "


def classify_delivery_detail(text: str) -> str:
    """Typed reason for a FAILED delivery turn, from the child's combined stdout+stderr.

    A one-shot CLI child reports a session-lease refusal with its own code on stderr
    (``hermes_cli.active_sessions.format_refusal_stderr``), and that refusal is a TARGET-STATE
    condition, not a turn failure: the target's chat has a live owner mid-turn, so the turn never
    ran. Naming it ``target_session_live`` is what lets the sender tell it apart from a held turn
    lock — both shipped ``target_busy`` before, and this lane did not even name the lease case
    (``unknown``). Everything else classifies from the raw text as usual, so older CLIs without
    the marker still resolve via the historical wording.
    """
    detail = str(text or "")
    for line in detail.splitlines():
        stripped = line.strip()
        if stripped.startswith(_REFUSAL_MARKER):
            code = stripped[len(_REFUSAL_MARKER):].strip()
            return TARGET_SESSION_LIVE if code == SESSION_NOT_OWNED_REASON else classify_agent_error(detail)
    if "already has a live owner" in detail:
        return TARGET_SESSION_LIVE
    return classify_agent_error(detail)


def delivery_failure_reason(error: BaseException) -> str:
    """The typed reason for a delivery refusal, kept inside the documented vocabulary.

    An exception's own ``reason`` is trusted only when it names a real code: ``TurnBusyError``
    carries ``target_busy``, but ``reason`` is also a stdlib attribute on ``ssl.SSLError`` and
    ``urllib.error.URLError``, and forwarding one of those would put free text where consumers
    expect a closed set. Anything else is classified like every other failure. Shared by the
    relay lane (``bot_relay.deliver``) and the local runner (``bot_mode_dm --run-delivery``).
    """
    # The delivery-refusal codes extend the structured refusal enum and predate ALL_REASONS,
    # which stays the agent/provider vocabulary the hosted-room event schema and the retry
    # policy are total over (see DELIVERY_REFUSAL_REASONS).
    supplied = str(getattr(error, "reason", "") or "").strip()
    if supplied in DELIVERY_REFUSAL_REASONS or supplied in ALL_REASONS:
        return supplied
    return classify_agent_error(str(error))
