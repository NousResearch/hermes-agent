"""Tests for the consecutive-denial circuit breaker in smart approvals.

After ``approvals.denial_breaker_threshold`` consecutive guardian DENY
verdicts in one session, the deny message returned to the model escalates
from "Do NOT retry" to a hard-stop CIRCUIT BREAKER instruction. Any
approval resets the tally. State is per-session and capped in size.

Follows the existing smart-approval mocking patterns from
tests/tools/test_execute_code_approval_cluster.py: monkeypatch
``_smart_approve`` / ``_get_approval_mode`` on the module and drive the
public guard entry points.
"""

from __future__ import annotations

import pytest

from tools import approval as A

BREAKER_MARKER = "CIRCUIT BREAKER:"


@pytest.fixture
def breaker_session(monkeypatch):
    """A clean gateway smart-mode session with the guardian forced to DENY.

    Uses the gateway path with a notify callback that resolves 'deny'
    (user denies the smart-DENY override) so the guard returns a definitive
    BLOCKED message — the channel the breaker text rides on.
    """
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "breaker-test-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    session_key = "breaker-test-session"
    token = A.set_current_session_key(session_key)
    A._reset_denials(session_key)
    with A._lock:
        A._permanent_approved.discard("breaker-test-danger")
        A._permanent_approved.discard("execute_code")
        A._session_approved.get(session_key, set()).discard("breaker-test-danger")
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


def _register_resolver(session_key: str, result):
    """Notify callback resolving the newest queued approval with *result*."""
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entries[-1].result = result
                entries[-1].event.set()
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


def _denied_terminal(command="dangerous thing"):
    return A.check_all_command_guards(command, "local")


def _denied_execute_code(code="print('x')"):
    return A.check_execute_code_guard(code, "local")


# ---------------------------------------------------------------------------
# (a) Two denials -> normal message; third -> breaker text present
# ---------------------------------------------------------------------------

def test_breaker_trips_on_third_consecutive_denial(breaker_session):
    _register_resolver(breaker_session, "deny")

    first = _denied_terminal("dangerous one")
    second = _denied_terminal("dangerous two")
    third = _denied_terminal("dangerous three")

    assert first["approved"] is False
    assert BREAKER_MARKER not in first["message"]
    assert second["approved"] is False
    assert BREAKER_MARKER not in second["message"]
    assert third["approved"] is False
    assert BREAKER_MARKER in third["message"]
    assert "3 consecutive commands were blocked" in third["message"]
    assert "STOP attempting variations" in third["message"]


# ---------------------------------------------------------------------------
# (b) An approval resets the tally
# ---------------------------------------------------------------------------

def test_approval_resets_tally(breaker_session, monkeypatch):
    _register_resolver(breaker_session, "deny")
    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")

    # Guardian approves the next command → tally resets.
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "approve")
    ok = _denied_terminal("benign command")
    assert ok["approved"] is True and ok.get("smart_approved") is True

    # Back to denials: the count restarts, so the next deny is #1, not #3.
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    after = _denied_terminal("dangerous again")
    assert after["approved"] is False
    assert BREAKER_MARKER not in after["message"]


def test_human_approval_resets_tally(breaker_session):
    _register_resolver(breaker_session, "deny")
    _denied_terminal("dangerous one")
    _denied_terminal("dangerous two")

    # User overrides the smart DENY (one-operation approval) → tally resets.
    _register_resolver(breaker_session, "once")
    ok = _denied_terminal("dangerous but user says yes")
    assert ok["approved"] is True and ok.get("user_approved") is True

    _register_resolver(breaker_session, "deny")
    after = _denied_terminal("dangerous again")
    assert after["approved"] is False
    assert BREAKER_MARKER not in after["message"]


# ---------------------------------------------------------------------------
# (c) Threshold 0 disables the breaker
# ---------------------------------------------------------------------------

def test_threshold_zero_disables_breaker(breaker_session, monkeypatch):
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 0)
    _register_resolver(breaker_session, "deny")
    for i in range(5):
        res = _denied_terminal(f"dangerous {i}")
        assert res["approved"] is False
        assert BREAKER_MARKER not in res["message"]


# ---------------------------------------------------------------------------
# (d) Tally is per-session — two session keys are independent
# ---------------------------------------------------------------------------

def test_tally_is_per_session(breaker_session):
    other = "breaker-other-session"
    A._reset_denials(other)
    try:
        assert A._record_denial(breaker_session) == 1
        assert A._record_denial(breaker_session) == 2
        # A different session starts from zero.
        assert A._record_denial(other) == 1
        # And its denial did not advance the first session's count.
        assert A._record_denial(breaker_session) == 3
        # Resetting one session leaves the other intact.
        A._reset_denials(breaker_session)
        assert A._record_denial(other) == 2
        assert A._record_denial(breaker_session) == 1
    finally:
        A._reset_denials(other)


# ---------------------------------------------------------------------------
# (e) BOTH call paths increment: terminal guard and execute_code guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deny_call", [_denied_terminal, _denied_execute_code],
                         ids=["terminal", "execute_code"])
def test_both_paths_increment_and_trip(breaker_session, deny_call):
    _register_resolver(breaker_session, "deny")
    for _ in range(2):
        res = deny_call()
        assert res["approved"] is False
        assert BREAKER_MARKER not in res["message"]
    tripped = deny_call()
    assert tripped["approved"] is False
    assert BREAKER_MARKER in tripped["message"]


def test_paths_share_one_session_tally(breaker_session):
    """Denials from the terminal and execute_code paths accumulate together."""
    _register_resolver(breaker_session, "deny")
    assert BREAKER_MARKER not in _denied_terminal("dangerous one")["message"]
    assert BREAKER_MARKER not in _denied_execute_code()["message"]
    tripped = _denied_terminal("dangerous three")
    assert BREAKER_MARKER in tripped["message"]


# ---------------------------------------------------------------------------
# Headless hard-deny path (no cli/gateway/ask override) also increments
# ---------------------------------------------------------------------------

def test_headless_smart_deny_increments_and_trips(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "0")
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_smart_approve", lambda _c, _d: "deny")
    monkeypatch.setattr(A, "_get_denial_breaker_threshold", lambda: 3)
    monkeypatch.setattr(A, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(
        A, "detect_dangerous_command",
        lambda command: (True, "headless-breaker-danger", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )
    # CLI-interactive path: the owner denies via the prompt callback.
    monkeypatch.setattr(A, "prompt_dangerous_approval",
                        lambda *args, **kwargs: "deny")

    session_key = "headless-breaker-session"
    token = A.set_current_session_key(session_key)
    A._reset_denials(session_key)
    with A._lock:
        A._permanent_approved.discard("headless-breaker-danger")
        A._session_approved.get(session_key, set()).discard(
            "headless-breaker-danger")
    try:
        first = A.check_all_command_guards("dangerous h1", "local")
        second = A.check_all_command_guards("dangerous h2", "local")
        third = A.check_all_command_guards("dangerous h3", "local")
        assert BREAKER_MARKER not in first["message"]
        assert BREAKER_MARKER not in second["message"]
        assert BREAKER_MARKER in third["message"]
    finally:
        A.reset_current_session_key(token)
        A._reset_denials(session_key)


# ---------------------------------------------------------------------------
# Eviction cap: the tally dict never grows past _DENIAL_TALLY_MAX_SESSIONS
# ---------------------------------------------------------------------------

def test_tally_evicts_oldest_sessions():
    with A._lock:
        saved = dict(A._denial_tally)
        A._denial_tally.clear()
    try:
        for i in range(A._DENIAL_TALLY_MAX_SESSIONS + 10):
            A._record_denial(f"evict-session-{i}")
        with A._lock:
            assert len(A._denial_tally) == A._DENIAL_TALLY_MAX_SESSIONS
            # Oldest entries were evicted, newest survive.
            assert "evict-session-0" not in A._denial_tally
            assert (
                f"evict-session-{A._DENIAL_TALLY_MAX_SESSIONS + 9}"
                in A._denial_tally
            )
    finally:
        with A._lock:
            A._denial_tally.clear()
            A._denial_tally.update(saved)
