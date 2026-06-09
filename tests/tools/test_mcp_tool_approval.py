"""Tests for the MCP first-invoke approval guard (#16462).

Covers:

  1. tools.approval.check_mcp_tool_guard — the decision matrix (config
     toggle, yolo/off bypass, headless/cron auto-approve, session and
     wildcard approval short-circuits, gateway approve/deny/timeout/
     missing-notify, CLI callback prompting, persistence scopes).
  2. tools.mcp_tool._make_tool_handler — a denied guard blocks the tool
     call BEFORE anything is dispatched to the MCP server, and the denial
     does not bump the server circuit breaker.
"""

from __future__ import annotations

import json

import pytest

from tools import approval as A


SERVER = "test-mcp-server"
TOOL = "run_query"
KEY = f"mcp:{SERVER}:{TOOL}"


def _clean_state(session_key: str):
    with A._lock:
        A._session_approved.pop(session_key, None)
        A._session_yolo.discard(session_key)
        A._permanent_approved.discard(KEY)
        A._permanent_approved.discard(f"mcp:{SERVER}:*")
        A._pending.pop(session_key, None)
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)


@pytest.fixture
def gw_session(monkeypatch):
    """A clean gateway session: HERMES_GATEWAY_SESSION set, a bound session
    key, and isolated gateway queues/callbacks. Yields the session_key."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_get_approval_config", lambda: {})
    monkeypatch.setattr(A, "save_permanent_allowlist", lambda _p: None)

    session_key = "mcp-guard-test-session"
    token = A.set_current_session_key(session_key)
    _clean_state(session_key)
    try:
        yield session_key
    finally:
        A.reset_current_session_key(token)
        _clean_state(session_key)


@pytest.fixture
def cli_session(monkeypatch):
    """A clean interactive-CLI session context. Yields the session_key."""
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_get_approval_config", lambda: {})
    monkeypatch.setattr(A, "save_permanent_allowlist", lambda _p: None)

    session_key = "mcp-guard-cli-session"
    token = A.set_current_session_key(session_key)
    _clean_state(session_key)
    try:
        yield session_key
    finally:
        A.reset_current_session_key(token)
        _clean_state(session_key)


def _register_resolver(session_key: str, result):
    """Register a gateway notify callback that immediately resolves the most
    recent queued approval entry with *result* (simulating a user response)."""
    def cb(_approval_data):
        with A._lock:
            entries = A._gateway_queues.get(session_key, [])
            if entries:
                entry = entries[-1]
                entry.result = result
                entry.event.set()
    with A._lock:
        A._gateway_notify_cbs[session_key] = cb


# ---------------------------------------------------------------------------
# 1. Decision matrix
# ---------------------------------------------------------------------------

def test_guard_disabled_via_config(gw_session, monkeypatch):
    monkeypatch.setattr(A, "_get_approval_config",
                        lambda: {"mcp_first_invoke": False})
    # Even with a denier registered, the toggle short-circuits.
    _register_resolver(gw_session, "deny")
    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "x"})
    assert res["approved"] is True


def test_guard_disabled_via_config_string_value(gw_session, monkeypatch):
    monkeypatch.setattr(A, "_get_approval_config",
                        lambda: {"mcp_first_invoke": "false"})
    _register_resolver(gw_session, "deny")
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True


def test_guard_mode_off_bypasses(gw_session, monkeypatch):
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "off")
    _register_resolver(gw_session, "deny")
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True


def test_guard_session_yolo_bypasses(gw_session):
    A.enable_session_yolo(gw_session)
    try:
        _register_resolver(gw_session, "deny")
        assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True
    finally:
        A.disable_session_yolo(gw_session)


def test_guard_headless_auto_approves(monkeypatch):
    # No approval surface → preserve auto-run, matching the terminal guard's
    # non-interactive contract.
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_get_approval_config", lambda: {})
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True


def test_guard_cron_auto_approves(monkeypatch):
    # Cron has no user present; first-invoke visibility has no value there.
    # MCP exposure for unattended profiles is governed by tools.include/exclude.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_get_approval_config", lambda: {})
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True


def test_guard_gateway_user_approves_once_is_one_shot(gw_session):
    _register_resolver(gw_session, "once")
    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "1"})
    assert res["approved"] is True
    assert res.get("user_approved") is True
    # One-shot: approval must NOT persist to future calls.
    assert A.is_approved(gw_session, KEY) is False


def test_guard_gateway_user_approves_session_persists(gw_session):
    _register_resolver(gw_session, "session")
    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "1"})
    assert res["approved"] is True
    assert A.is_approved(gw_session, KEY) is True
    # Subsequent calls auto-approve without prompting (denier would fire).
    _register_resolver(gw_session, "deny")
    res2 = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "2"})
    assert res2["approved"] is True


def test_guard_gateway_user_approves_always_persists(gw_session):
    _register_resolver(gw_session, "always")
    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "1"})
    assert res["approved"] is True
    with A._lock:
        assert KEY in A._permanent_approved


def test_guard_session_approval_is_per_tool(gw_session):
    # Approving one tool must not approve a different tool on the same server.
    A.approve_session(gw_session, KEY)
    _register_resolver(gw_session, "deny")
    res = A.check_mcp_tool_guard(SERVER, "send_email", {})
    assert res["approved"] is False


def test_guard_server_wildcard_approves_all_tools(gw_session):
    A.approve_session(gw_session, f"mcp:{SERVER}:*")
    _register_resolver(gw_session, "deny")
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True
    assert A.check_mcp_tool_guard(SERVER, "send_email", {})["approved"] is True


def test_guard_clear_session_resets_approval(gw_session):
    _register_resolver(gw_session, "session")
    assert A.check_mcp_tool_guard(SERVER, TOOL, {})["approved"] is True
    assert A.is_approved(gw_session, KEY) is True
    A.clear_session(gw_session)
    assert A.is_approved(gw_session, KEY) is False


def test_guard_gateway_user_denies_blocks(gw_session):
    _register_resolver(gw_session, "deny")
    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "1"})
    assert res["approved"] is False
    assert res["outcome"] == "denied"
    assert res["user_consent"] is False
    assert "Do NOT retry" in res["message"]


def test_guard_gateway_timeout_blocks(gw_session, monkeypatch):
    # Callback that never resolves; force an immediate timeout.
    with A._lock:
        A._gateway_notify_cbs[gw_session] = lambda _d: None
    monkeypatch.setattr(A, "_get_approval_config",
                        lambda: {"gateway_timeout": 0})
    res = A.check_mcp_tool_guard(SERVER, TOOL, {})
    assert res["approved"] is False
    assert res["outcome"] == "timeout"
    assert "Silence is not consent" in res["message"]


def test_guard_gateway_missing_notify_is_pending(gw_session):
    res = A.check_mcp_tool_guard(SERVER, TOOL, {})
    assert res["approved"] is False
    assert res["status"] == "pending_approval"
    assert res["pattern_key"] == KEY


def test_guard_prompt_includes_server_tool_and_args(gw_session):
    seen = {}

    def cb(approval_data):
        seen.update(approval_data)
        with A._lock:
            entries = A._gateway_queues.get(gw_session, [])
            if entries:
                entries[-1].result = "once"
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[gw_session] = cb

    A.check_mcp_tool_guard(SERVER, TOOL, {"query": "DROP TABLE users"})
    assert SERVER in seen["command"]
    assert TOOL in seen["command"]
    assert "DROP TABLE users" in seen["command"]
    assert seen["pattern_key"] == KEY


def test_guard_args_preview_truncated(gw_session):
    seen = {}

    def cb(approval_data):
        seen.update(approval_data)
        with A._lock:
            entries = A._gateway_queues.get(gw_session, [])
            if entries:
                entries[-1].result = "once"
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[gw_session] = cb

    A.check_mcp_tool_guard(SERVER, TOOL, {"blob": "x" * 5000})
    assert len(seen["command"]) < 1000
    assert "truncated" in seen["command"]


# ---------------------------------------------------------------------------
# 2. CLI interactive path
# ---------------------------------------------------------------------------

def test_guard_cli_callback_deny_blocks(cli_session):
    calls = {}

    def cb(command, description, allow_permanent=True):
        calls["command"] = command
        calls["description"] = description
        return "deny"

    res = A.check_mcp_tool_guard(SERVER, TOOL, {"q": "x"},
                                 approval_callback=cb)
    assert res["approved"] is False
    assert res["outcome"] == "denied"
    assert SERVER in calls["command"]
    assert TOOL in calls["command"]


def test_guard_cli_callback_session_persists(cli_session):
    res = A.check_mcp_tool_guard(
        SERVER, TOOL, {},
        approval_callback=lambda c, d, allow_permanent=True: "session")
    assert res["approved"] is True
    assert A.is_approved(cli_session, KEY) is True
    # Second call short-circuits without prompting.
    def explode(*_a, **_k):
        raise AssertionError("must not re-prompt after session approval")
    res2 = A.check_mcp_tool_guard(SERVER, TOOL, {}, approval_callback=explode)
    assert res2["approved"] is True


# ---------------------------------------------------------------------------
# 3. Handler integration — deny blocks before MCP dispatch
# ---------------------------------------------------------------------------

def test_handler_blocks_before_dispatch_when_guard_denies(gw_session, monkeypatch):
    import tools.mcp_tool as M

    class _FakeServer:
        session = object()  # truthy: passes the connected check

    monkeypatch.setitem(M._servers, SERVER, _FakeServer())
    monkeypatch.setitem(M._server_error_counts, SERVER, 0)

    def _no_dispatch(*_a, **_k):
        raise AssertionError("MCP loop must not be reached on guard deny")
    monkeypatch.setattr(M, "_run_on_mcp_loop", _no_dispatch)

    _register_resolver(gw_session, "deny")
    handler = M._make_tool_handler(SERVER, TOOL, tool_timeout=5)
    result = json.loads(handler({"q": "x"}))
    assert "error" in result
    assert "BLOCKED" in result["error"]
    # User denial is not a server fault — breaker must not be bumped.
    assert M._server_error_counts.get(SERVER, 0) == 0


def test_handler_dispatches_when_guard_approves(gw_session, monkeypatch):
    import tools.mcp_tool as M

    class _FakeServer:
        session = object()

    monkeypatch.setitem(M._servers, SERVER, _FakeServer())
    monkeypatch.setitem(M._server_error_counts, SERVER, 0)
    monkeypatch.setattr(
        M, "_run_on_mcp_loop",
        lambda coro_or_factory, timeout=30: json.dumps({"result": "ok"}))

    _register_resolver(gw_session, "once")
    handler = M._make_tool_handler(SERVER, TOOL, tool_timeout=5)
    result = json.loads(handler({"q": "x"}))
    assert result == {"result": "ok"}
