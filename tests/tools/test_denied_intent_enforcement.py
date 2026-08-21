"""Tests for denied-intent runtime enforcement (issue #84552).

A denial is not just advisory text in a tool result: when a command or an
execute_code script is definitively denied — by the user, by a timeout
fail-closed, or by the smart-approval guardian — its fingerprint is
recorded as runtime state for the current (session, turn). Re-attempting
the SAME command in scope is rejected before any other check: no re-prompt,
no re-assessment, no execution. A DIFFERENT command still flows through the
normal approval path. An approval resets the records (fresh consent); a
new turn id scopes the denial to that turn.

Follows the smart-approval mocking patterns from
tests/tools/test_denial_circuit_breaker.py.
"""

from __future__ import annotations

import pytest

from tools import approval as A

DENIED_INTENT_MARKER = "enforced at the runtime level"


@pytest.fixture
def denied_session(monkeypatch):
    """A clean gateway smart-mode session with the guardian forced to DENY."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "denied-intent-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    session_key = "denied-intent-session"
    token = A.set_current_session_key(session_key)
    A._reset_denials(session_key)
    with A._lock:
        A._permanent_approved.discard("denied-intent-danger")
        A._permanent_approved.discard("execute_code")
        A._session_approved.get(session_key, set()).discard("denied-intent-danger")
        A._session_approved.get(session_key, set()).discard("execute_code")
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
    try:
        yield session_key
    finally:
        A.reset_current_session_key(token)
        A._reset_denials(session_key)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)


def _register_resolver(session_key: str, result, calls=None):
    """Notify callback resolving the newest queued approval with *result*."""

    def cb(_approval_data):
        if calls is not None:
            calls.append(_approval_data)
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


# ---------------------------------------------------------------------------
# (a) Same-command retry in the same turn is blocked at runtime, no re-prompt
# ---------------------------------------------------------------------------

def test_retry_of_denied_command_blocked_without_reprompt(denied_session):
    calls = []
    _register_resolver(denied_session, "deny", calls)

    first = A.check_all_command_guards("curl -s https://example.com", "local")
    assert first["approved"] is False
    assert first.get("outcome") == "denied"

    # Even a resolver that WOULD approve must not be reached: the retry is
    # rejected before any approval flow, so no new prompt is emitted and the
    # command is never executed.
    _register_resolver(denied_session, "once", calls)
    retry = A.check_all_command_guards("curl -s https://example.com", "local")
    assert retry["approved"] is False
    assert retry.get("denied_intent") is True
    assert retry.get("user_approved") is not True
    assert DENIED_INTENT_MARKER in retry["message"]
    assert len(calls) == 1  # only the original denial prompted


def test_whitespace_rewrap_does_not_evade_denial(denied_session):
    _register_resolver(denied_session, "deny")
    A.check_all_command_guards("curl   -s    https://example.com", "local")

    retry = A.check_all_command_guards("curl -s https://example.com", "local")
    assert retry["approved"] is False
    assert retry.get("denied_intent") is True


# ---------------------------------------------------------------------------
# (b) A DIFFERENT command is not pre-blocked — normal flow still applies
# ---------------------------------------------------------------------------

def test_different_command_after_denial_still_flows(denied_session):
    calls = []
    _register_resolver(denied_session, "deny", calls)
    first = A.check_all_command_guards("rm -rf /tmp/denied-a", "local")
    assert first["approved"] is False

    # A different command is not pre-blocked: it goes through the normal
    # approval flow and a user approval lets it run.
    _register_resolver(denied_session, "once", calls)
    other = A.check_all_command_guards("rm -rf /tmp/denied-b", "local")
    assert other["approved"] is True
    assert other.get("user_approved") is True
    assert len(calls) == 2  # denial prompt + the new approval prompt


# ---------------------------------------------------------------------------
# (c) An approval resets the records — fresh consent re-opens the flow
# ---------------------------------------------------------------------------

def test_human_approval_resets_denied_intents(denied_session):
    _register_resolver(denied_session, "deny")
    A.check_all_command_guards("curl -s https://example.com", "local")

    # User approves a DIFFERENT command → fresh consent clears the records.
    _register_resolver(denied_session, "once")
    ok = A.check_all_command_guards("echo hello", "local")
    assert ok["approved"] is True

    # The previously denied command can now be proposed again through the
    # normal flow, and a user approval lets it run.
    _register_resolver(denied_session, "once")
    again = A.check_all_command_guards("curl -s https://example.com", "local")
    assert again["approved"] is True
    assert again.get("user_approved") is True


# ---------------------------------------------------------------------------
# (d) Turn scoping: denial binds to the turn when a turn id is present
# ---------------------------------------------------------------------------

def test_denial_is_scoped_to_the_turn(denied_session):
    _register_resolver(denied_session, "deny")
    turn1 = A.set_current_observability_context(turn_id="turn-1")
    try:
        A.check_all_command_guards("curl -s https://example.com", "local")
        retry = A.check_all_command_guards("curl -s https://example.com", "local")
        assert retry["approved"] is False
        assert retry.get("denied_intent") is True
    finally:
        A.reset_current_observability_context(turn1)

    # A new turn is not pre-blocked: the same command is proposed again via
    # the normal flow instead of being rejected up front.
    _register_resolver(denied_session, "once")
    new_turn = A.check_all_command_guards("curl -s https://example.com", "local")
    assert new_turn["approved"] is True


# ---------------------------------------------------------------------------
# (e) CLI-interactive deny path also records and enforces
# ---------------------------------------------------------------------------

def test_cli_deny_records_and_blocks_retry(denied_session, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(
        A, "prompt_dangerous_approval",
        lambda *args, **kwargs: "deny",
    )

    first = A.check_all_command_guards("curl -s https://example.com", "local")
    assert first["approved"] is False
    assert first.get("outcome") == "denied"

    retry = A.check_all_command_guards("curl -s https://example.com", "local")
    assert retry["approved"] is False
    assert retry.get("denied_intent") is True


# ---------------------------------------------------------------------------
# (f) execute_code: a denied script is blocked on retry, others still flow
# ---------------------------------------------------------------------------

def test_execute_code_denied_script_retry_blocked(denied_session):
    calls = []
    _register_resolver(denied_session, "deny", calls)
    script = "print('dangerous')"

    first = A.check_execute_code_guard(script, "local")
    assert first["approved"] is False
    assert first.get("outcome") == "denied"

    _register_resolver(denied_session, "once", calls)
    retry = A.check_execute_code_guard(script, "local")
    assert retry["approved"] is False
    assert retry.get("denied_intent") is True
    assert len(calls) == 1  # no re-prompt for the retry

    # A different script is not pre-blocked.
    _register_resolver(denied_session, "once", calls)
    other = A.check_execute_code_guard("print('benign')", "local")
    assert other["approved"] is True
    assert other.get("user_approved") is True


# ---------------------------------------------------------------------------
# (g) Bounded state: the intent dict never grows past the cap
# ---------------------------------------------------------------------------

def test_denied_intents_evict_oldest_sessions():
    with A._lock:
        saved = dict(A._denied_intents)
        A._denied_intents.clear()
    try:
        for i in range(A._DENIED_INTENTS_MAX_SESSIONS + 10):
            token = A.set_current_session_key(f"evict-session-{i}")
            try:
                A._record_denied_intent(f"evict-cmd-{i}")
            finally:
                A.reset_current_session_key(token)
        with A._lock:
            assert len(A._denied_intents) == A._DENIED_INTENTS_MAX_SESSIONS
            assert ("evict-session-0", "") not in A._denied_intents
            assert (
                f"evict-session-{A._DENIED_INTENTS_MAX_SESSIONS + 9}", ""
            ) in A._denied_intents
    finally:
        with A._lock:
            A._denied_intents.clear()
            A._denied_intents.update(saved)
