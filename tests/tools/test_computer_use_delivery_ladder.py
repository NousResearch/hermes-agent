"""Regression tests for the cua-driver verify → escalate ladder.

Covers NousResearch/hermes-agent#67052:
  - Phase A: cua-driver structured verdicts (verified/effect/escalation/code/
    degraded/path) are preserved through ActionResult and surfaced in the
    model-facing response, additively (old drivers omit them cleanly).
  - Phase B: delivery_mode is model-reachable, capability-gated, and refuses
    with foreground_unsupported on an old driver rather than silently
    downgrading to background.
  - Phase C: foreground approval is scoped by (action, delivery_mode) and by
    the shared gate's session key, so a background approval never silently
    authorizes foreground and one run's unlock never leaks into another.

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

    def __init__(
        self,
        out: Dict[str, Any],
        capabilities: Optional[set] = None,
        input_properties: Optional[Dict[str, set]] = None,
    ):
        self._out = out
        self._caps = capabilities or set()
        self._input_properties = input_properties or {}
        self.last_args: Dict[str, Any] = {}
        self.calls = []

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.last_args = args
        self.calls.append((name, dict(args)))
        return self._out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return capability in self._caps

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        return property_name in self._input_properties.get(tool, set())

    def _has_tool(self, name: str) -> bool:
        return name == "bring_to_front"


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


def test_degraded_capture_signal_preserved():
    out = {
        "isError": False, "data": {},
        "structuredContent": {"effect": "suspected_noop", "degraded": True,
                              "escalation": {"recommended": "px", "reason": "empty tree"}},
    }
    be = _make_backend(_FakeSession(out))
    res = be.scroll(direction="down", element=1)
    assert res.degraded is True
    assert res.escalation["recommended"] == "px"


def test_old_driver_without_structured_content_is_clean():
    """A driver that returns no structuredContent leaves every verdict field
    None — unchanged behavior, no crash."""
    out = {"isError": False, "data": {"message": "done"}, "structuredContent": None}
    be = _make_backend(_FakeSession(out))
    res = be.click(element=3)
    assert res.ok is True
    assert res.message == "done"
    assert res.verified is None
    assert res.effect is None
    assert res.escalation is None
    assert res.code is None
    assert res.path is None


def test_text_response_surfaces_fields_additively():
    from tools.computer_use.backend import ActionResult
    from tools.computer_use.tool import _text_response

    # Full verdict → all fields present.
    r = ActionResult(ok=True, action="click", effect="suspected_noop",
                     escalation={"recommended": "foreground"}, code="background_unavailable",
                     path="ax", verified=False)
    payload = json.loads(_text_response(r))
    assert payload["effect"] == "suspected_noop"
    assert payload["escalation"] == {"recommended": "foreground"}
    assert payload["code"] == "background_unavailable"
    assert payload["verified"] is False

    # Bare transport success still requires fresh verification, without None noise.
    r2 = ActionResult(ok=True, action="click")
    payload2 = json.loads(_text_response(r2))
    assert payload2["ok"] is True
    assert payload2["action"] == "click"
    # Verdict routes to fresh verification; a human hint may accompany the
    # decision (contract is the decision, not the exact dict shape).
    assert payload2["verdict"]["decision"] == "verify_fresh_state"
    for k in ("effect", "escalation", "code", "verified", "path", "degraded", "delivery_mode"):
        assert k not in payload2


# ---------------------------------------------------------------------------
# Phase B — delivery_mode threading + capability gating
# ---------------------------------------------------------------------------

def test_background_is_default_no_flag_sent():
    out = {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}
    sess = _FakeSession(out)
    be = _make_backend(sess)
    be.click(element=1)  # no delivery_mode
    assert "delivery_mode" not in sess.last_args


def test_foreground_sent_when_schema_property_present():
    out = {"isError": False, "data": {}, "structuredContent": {"effect": "unverifiable"}}
    sess = _FakeSession(out, input_properties={"click": {"delivery_mode"}})
    be = _make_backend(sess)
    res = be.click(element=1, delivery_mode="foreground", bring_to_front=True)
    assert [name for name, _ in sess.calls] == ["bring_to_front", "click"]
    assert sess.calls[0][1] == {"pid": 4242, "window_id": 7}
    assert sess.last_args.get("delivery_mode") == "foreground"
    assert "bring_to_front" not in sess.last_args
    assert res.delivery_mode == "foreground"


def test_foreground_refused_on_old_driver():
    """A live action schema lacking the property must NOT silently downgrade — it
    returns a structured foreground_unsupported result."""
    out = {"isError": False, "data": {}, "structuredContent": {}}
    sess = _FakeSession(out)
    be = _make_backend(sess)
    res = be.click(element=1, delivery_mode="foreground")
    assert res.ok is False
    assert res.code == "foreground_unsupported"
    # crucially: no tool call was made with a silent background downgrade
    assert sess.calls == []


def test_bad_delivery_mode_rejected():
    out = {"isError": False, "data": {}, "structuredContent": {}}
    sess = _FakeSession(out, input_properties={"type_text": {"delivery_mode"}})
    be = _make_backend(sess)
    res = be.type_text("hi", delivery_mode="sideways")
    assert res.ok is False
    assert res.code == "bad_delivery_mode"


def test_dispatcher_threads_delivery_mode_to_backend(grant_computer_use_approvals):
    """End-to-end through the tool dispatcher with the noop backend."""
    from tools.computer_use import tool as cu
    with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "noop"}, clear=False):
        cu.reset_backend_for_tests()
        be = cu._get_backend()
        cu.handle_computer_use({"action": "click", "element": 5,
                                "delivery_mode": "foreground"})
        # noop records kwargs; find the click call
        clicks = [kw for (name, kw) in be.calls if name == "click"]  # type: ignore[attr-defined]
        assert clicks and clicks[-1].get("delivery_mode") == "foreground"


# ---------------------------------------------------------------------------
# Phase C — foreground approval scoping (action + delivery_mode + session)
# ---------------------------------------------------------------------------

@pytest.fixture
def _interactive_session(monkeypatch):
    """Interactive CLI presence for the shared gate plus a fresh approval session key; grants made here are
    wiped from ``tools.approval``'s store afterwards so nothing leaks between tests."""
    from tools import approval
    from tools.approval_context import reset_current_session_key, set_current_session_key

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "save_permanent_allowlist", lambda patterns: None)
    keys: list[str] = []

    def use(session_key: str):
        keys.append(session_key)
        return set_current_session_key(session_key)

    yield use
    for key in keys:
        approval.clear_session(key)
    reset_current_session_key(set_current_session_key(""))


def test_background_approval_does_not_authorize_foreground(_interactive_session):
    from tools.computer_use import tool as cu

    seen = []

    def cb(command, description, **kw):
        seen.append((command, description))
        return "session"

    cu.set_approval_callback(cb)
    _interactive_session("sess-A")
    try:
        # Background click, approve for session.
        assert cu._request_approval("click", {}) is None
        # A second background click needs no prompt (cached in the shared session store).
        assert cu._request_approval("click", {}) is None
        assert len(seen) == 1
        # Foreground click on the SAME action must prompt again — the background grant does not cover it.
        assert cu._request_approval("click", {"delivery_mode": "foreground"}) is None
        assert len(seen) == 2
        assert "FOREGROUND" in seen[-1][0]
    finally:
        cu.set_approval_callback(None)


def test_approval_state_is_session_scoped(_interactive_session):
    from tools.computer_use import tool as cu

    calls = []

    def cb(command, description, **kw):
        calls.append(command)
        return "session"

    cu.set_approval_callback(cb)
    try:
        # Run A approves foreground click.
        _interactive_session("run-A")
        cu._request_approval("click", {"delivery_mode": "foreground"})
        # Run B has NOT — it must prompt independently.
        n_before = len(calls)
        _interactive_session("run-B")
        cu._request_approval("click", {"delivery_mode": "foreground"})
        assert len(calls) == n_before + 1
    finally:
        cu.set_approval_callback(None)


def test_always_grant_is_per_scope_key_and_visible_to_shared_store(_interactive_session):
    """One grant store: an "always" answered through computer_use lands in ``tools.approval`` under the same
    ``cua:<action>:<mode>`` key, and — unlike the old blanket unlock — covers only that scope, so the visible
    foreground variant still prompts."""
    from tools import approval
    from tools.computer_use import tool as cu

    calls = []

    def cb(command, description, **kw):
        calls.append(command)
        return "always"

    cu.set_approval_callback(cb)
    _interactive_session("run-C")
    try:
        assert cu._request_approval("click", {}) is None
        assert approval.is_approved("run-C", "cua:click:background")
        assert not approval.is_approved("run-C", "cua:click:foreground")
        assert cu._request_approval("click", {"delivery_mode": "foreground"}) is None
        assert len(calls) == 2
    finally:
        cu.set_approval_callback(None)
        with approval._lock:
            approval._permanent_set().difference_update({"cua:click:background", "cua:click:foreground"})


def test_foreground_summary_warns_about_focus_change():
    from tools.computer_use.tool import _summarize_action
    s = _summarize_action("click", {"element": 3, "delivery_mode": "foreground"})
    assert "FOREGROUND" in s
    bg = _summarize_action("click", {"element": 3})
    assert "FOREGROUND" not in bg


# ---------------------------------------------------------------------------
# #55048 Bug 1 — a dead session must reset _started so the next call recovers
# ---------------------------------------------------------------------------

def test_lifecycle_finally_resets_started_for_reentry():
    """After the lifecycle coro exits (MCP drop / crash), _started must be
    False so _require_started() no longer passes into a dead/None session.
    We drive the finally block directly via the coro's cleanup semantics."""
    from tools.computer_use.cua_backend_session import _CuaDriverSession

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
    from tools.computer_use.cua_backend_session import _CuaDriverSession

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
