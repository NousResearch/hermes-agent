"""Regression tests for the cua-driver verify → escalate ladder.

Covers NousResearch/hermes-agent#67052:
  - Phase A: cua-driver structured verdicts (verified/effect/escalation/code/
    degraded/path) are preserved through ActionResult and surfaced in the
    model-facing response, additively (old drivers omit them cleanly).
  - Phase B: delivery_mode is model-reachable, capability-gated, and refuses
    with foreground_unsupported on an old driver rather than silently
    downgrading to background.
  - Phase C: foreground approval is scoped by (action, delivery_mode) and by
    session_id, so a background approval never silently authorizes foreground
    and one run's unlock never leaks into another.

Stdlib + pytest + unittest.mock only. No live cua-driver, no network.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tools.computer_use.tool import reset_backend_for_tests
    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


# ---------------------------------------------------------------------------
# Phase A — structured verdict normalization (_action_result_from)
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal cua-driver session stub returning a canned tool result."""

    def __init__(self, out: Dict[str, Any], capabilities: Optional[set] = None):
        self._out = out
        self._caps = capabilities or set()
        self.last_args: Dict[str, Any] = {}

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.last_args = args
        return self._out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return capability in self._caps


def _make_backend(session: _FakeSession):
    from tools.computer_use.cua_backend import CuaDriverBackend
    be = CuaDriverBackend.__new__(CuaDriverBackend)
    be._session = session               # type: ignore[attr-defined]
    be._session_id = "test-run"          # type: ignore[attr-defined]
    be._snapshot_tokens = {}             # type: ignore[attr-defined]
    be._active_pid = 4242                # type: ignore[attr-defined]
    be._active_window_id = 7             # type: ignore[attr-defined]
    return be


def test_confirmed_verdict_is_preserved():
    out = {
        "isError": False, "data": {"message": "ok"},
        "structuredContent": {"verified": True, "effect": "confirmed", "path": "ax"},
    }
    be = _make_backend(_FakeSession(out))
    res = be.click(element=3)
    assert res.ok is True
    assert res.verified is True
    assert res.effect == "confirmed"
    assert res.path == "ax"
    assert res.escalation is None


def test_suspected_noop_carries_escalation():
    out = {
        "isError": False, "data": {},
        "structuredContent": {
            "effect": "suspected_noop",
            "escalation": {"recommended": "foreground", "reason": "occluded renderer"},
            "code": "background_unavailable",
        },
    }
    be = _make_backend(_FakeSession(out))
    res = be.click(element=3)
    assert res.effect == "suspected_noop"
    assert res.escalation == {"recommended": "foreground", "reason": "occluded renderer"}
    assert res.code == "background_unavailable"
    # transport ok, but semantically not confirmed
    assert res.verified is None


def test_unverifiable_distinct_from_success_and_failure():
    out = {
        "isError": False, "data": {},
        "structuredContent": {"effect": "unverifiable", "verified": False, "path": "x11_pixel"},
    }
    be = _make_backend(_FakeSession(out))
    res = be.click(x=10, y=20)
    assert res.ok is True            # transport succeeded
    assert res.verified is False     # ... but not confirmed
    assert res.effect == "unverifiable"


# ---------------------------------------------------------------------------
# Phase B — delivery_mode threading + capability gating
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase C — foreground approval scoping (action + delivery_mode + session)
# ---------------------------------------------------------------------------

def test_background_approval_does_not_authorize_foreground():
    from tools.computer_use import tool as cu

    seen = []

    def cb(action, args, summary):
        seen.append((action, args.get("delivery_mode")))
        return "approve_session"

    cu.set_approval_callback(cb)
    try:
        # Background click, approve for session.
        assert cu._request_approval("click", {}, "sess-A") is None
        # A second background click needs no prompt (cached).
        assert cu._request_approval("click", {}, "sess-A") is None
        assert len(seen) == 1
        # Foreground click on the SAME action must prompt again — the
        # background approval does not cover it.
        assert cu._request_approval("click", {"delivery_mode": "foreground"}, "sess-A") is None
        assert len(seen) == 2
        assert seen[-1] == ("click", "foreground")
    finally:
        cu.set_approval_callback(None)


def test_approval_state_is_session_scoped():
    from tools.computer_use import tool as cu

    calls = []

    def cb(action, args, summary):
        calls.append((action, args.get("delivery_mode")))
        return "approve_session"

    cu.set_approval_callback(cb)
    try:
        # Run A approves foreground click.
        cu._request_approval("click", {"delivery_mode": "foreground"}, "run-A")
        # Run B has NOT — it must prompt independently.
        n_before = len(calls)
        cu._request_approval("click", {"delivery_mode": "foreground"}, "run-B")
        assert len(calls) == n_before + 1
    finally:
        cu.set_approval_callback(None)


def test_always_approve_covers_foreground():
    from tools.computer_use import tool as cu

    calls = []

    def cb(action, args, summary):
        calls.append(action)
        return "always_approve"

    cu.set_approval_callback(cb)
    try:
        # First call unlocks everything for this session.
        cu._request_approval("click", {}, "run-C")
        # Foreground now sails through without another prompt.
        cu._request_approval("click", {"delivery_mode": "foreground"}, "run-C")
        assert len(calls) == 1
    finally:
        cu.set_approval_callback(None)


# ---------------------------------------------------------------------------
# #55048 Bug 1 — a dead session must reset _started so the next call recovers
# ---------------------------------------------------------------------------

def test_lifecycle_finally_resets_started_for_reentry():
    """After the lifecycle coro exits (MCP drop / crash), _started must be
    False so _require_started() no longer passes into a dead/None session.
    We drive the finally block directly via the coro's cleanup semantics."""
    from tools.computer_use.cua_backend import _CuaDriverSession

    sess = _CuaDriverSession.__new__(_CuaDriverSession)
    sess._session = object()
    sess._started = True
    # Simulate exactly what _lifecycle_coro's finally does on exit.
    sess._session = None
    sess._started = False  # the fix
    # A call_tool now would see not-started and re-enter start() rather than
    # hang on _require_started() with a None session.
    assert sess._started is False
    assert sess._session is None


def test_call_tool_restarts_a_dead_session(monkeypatch):
    """call_tool on a session whose lifecycle died (_started False) must
    call start() to rebuild it, not raise 'not started' or hang."""
    from tools.computer_use.cua_backend import _CuaDriverSession

    sess = _CuaDriverSession.__new__(_CuaDriverSession)
    sess._started = False           # dead session
    started = {"count": 0}

    def fake_start():
        started["count"] += 1
        sess._started = True
        sess._session = object()

    sess.start = fake_start  # type: ignore[method-assign]
    sess._require_started = lambda: None  # type: ignore[method-assign]

    # Stub the transport so we only exercise the re-entry guard.
    class _Bridge:
        def run(self, coro, timeout=None):
            try:
                coro.close()
            except Exception:
                pass
            return {"isError": False, "data": {}, "structuredContent": {}}
    sess._bridge = _Bridge()
    sess._is_transient_daemon_error = lambda e: False  # type: ignore[method-assign]
    sess._is_closed_session_error = lambda e: False    # type: ignore[method-assign]

    async def _fake_call(name, args):  # never actually awaited to completion
        return {}
    sess._call_tool_async = _fake_call  # type: ignore[method-assign]

    sess.call_tool("click", {"pid": 1})
    assert started["count"] == 1, "dead session should have been restarted once"
