"""Unit tests for the supervisor-WS fast path in browser_console / _browser_eval.

These exercise the dispatch logic in ``tools.browser_tool._browser_eval`` and
the response shaping in ``CDPSupervisor.evaluate_runtime`` using mocks — no
real browser, no real WebSocket.  Real-CDP coverage lives in
``tests/tools/test_browser_supervisor.py`` (gated on Chrome being installed).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fast-path dispatch: tools.browser_tool._browser_eval
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_camofox(monkeypatch):
    """Force the non-camofox path so our supervisor branch is reached."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: "test-task")


def _patch_supervisor(monkeypatch, supervisor):
    """Wire SUPERVISOR_REGISTRY.get to return ``supervisor`` for any task_id."""
    import tools.browser_supervisor as bs

    registry = MagicMock()
    registry.get.return_value = supervisor
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    return registry


class TestBrowserEvalSupervisorPath:
    """The supervisor fast path replaces the agent-browser subprocess hop."""

    def test_primitive_result_routes_through_supervisor(self, monkeypatch):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": True,
            "result": 42,
            "result_type": "number",
        }
        _patch_supervisor(monkeypatch, sup)
        # If the subprocess path is hit we want a loud failure.
        monkeypatch.setattr(
            bt, "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess path must not run when supervisor is healthy"),
        )

        out = json.loads(bt._browser_eval("1 + 41"))
        assert out["success"] is True
        assert out["result"] == 42
        assert out["method"] == "cdp_supervisor"
        sup.evaluate_runtime.assert_called_once_with("1 + 41")

    def test_json_string_result_is_parsed(self, monkeypatch):
        """Match agent-browser semantics: JSON-string results get parsed."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": True,
            "result": '{"a": 1, "b": [2, 3]}',
            "result_type": "string",
        }
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setattr(
            bt, "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess path must not run"),
        )

        out = json.loads(bt._browser_eval('JSON.stringify({a:1,b:[2,3]})'))
        assert out["success"] is True
        assert out["result"] == {"a": 1, "b": [2, 3]}
        # result_type reflects the parsed Python type, not the raw JS type.
        assert out["result_type"] == "dict"

    def test_non_json_string_result_kept_as_string(self, monkeypatch):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": True,
            "result": "hello world",
            "result_type": "string",
        }
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setattr(bt, "_run_browser_command", lambda *a, **kw: pytest.fail("nope"))

        out = json.loads(bt._browser_eval('"hello world"'))
        assert out["result"] == "hello world"
        assert out["result_type"] == "str"

    def test_js_exception_surfaces_without_subprocess_fallthrough(self, monkeypatch):
        """A JS-side error must NOT trigger a (slow + redundant) subprocess retry."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": False,
            "error": "Uncaught ReferenceError: foo is not defined",
        }
        _patch_supervisor(monkeypatch, sup)
        called = {"subprocess": False}

        def _fake_subprocess(*a, **kw):
            called["subprocess"] = True
            return {"success": True, "data": {"result": "should-not-be-used"}}

        monkeypatch.setattr(bt, "_run_browser_command", _fake_subprocess)

        out = json.loads(bt._browser_eval("foo.bar"))
        assert out["success"] is False
        assert "ReferenceError" in out["error"]
        assert called["subprocess"] is False, \
            "JS exception should be surfaced, not retried via subprocess"

    def test_supervisor_loop_down_falls_through_to_subprocess(self, monkeypatch):
        """When the supervisor itself is unavailable, fall back to the subprocess."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": False,
            "error": "supervisor loop is not running",
        }
        _patch_supervisor(monkeypatch, sup)

        called = {"subprocess": False}

        def _fake_subprocess(task_id, cmd, args):
            called["subprocess"] = True
            assert cmd == "eval"
            return {"success": True, "data": {"result": "fallback-result"}}

        monkeypatch.setattr(bt, "_run_browser_command", _fake_subprocess)

        out = json.loads(bt._browser_eval("anything"))
        assert called["subprocess"] is True
        assert out["success"] is True
        assert out["result"] == "fallback-result"
        # Subprocess path doesn't tag the response with method=cdp_supervisor.
        assert out.get("method") != "cdp_supervisor"

    def test_no_active_supervisor_falls_through_to_subprocess(self, monkeypatch):
        """When SUPERVISOR_REGISTRY.get returns None, subprocess path runs."""
        import tools.browser_tool as bt

        _patch_supervisor(monkeypatch, None)
        called = {"subprocess": False}

        def _fake_subprocess(task_id, cmd, args):
            called["subprocess"] = True
            return {"success": True, "data": {"result": "agent-browser-result"}}

        monkeypatch.setattr(bt, "_run_browser_command", _fake_subprocess)

        out = json.loads(bt._browser_eval("1+1"))
        assert called["subprocess"] is True
        assert out["success"] is True
        assert out.get("method") != "cdp_supervisor"

    def test_supervisor_no_session_falls_through(self, monkeypatch):
        """A supervisor without an attached page session must fall through cleanly."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": False,
            "error": "supervisor has no attached page session",
        }
        _patch_supervisor(monkeypatch, sup)
        called = {"subprocess": False}

        def _fake_subprocess(*a, **kw):
            called["subprocess"] = True
            return {"success": True, "data": {"result": "fallback"}}

        monkeypatch.setattr(bt, "_run_browser_command", _fake_subprocess)
        json.loads(bt._browser_eval("1+1"))
        assert called["subprocess"] is True


# ---------------------------------------------------------------------------
# Response shaping: CDPSupervisor.evaluate_runtime
# ---------------------------------------------------------------------------


def _make_supervisor_with_cdp(cdp_response):
    """Build a CDPSupervisor instance that mocks ``_cdp`` to return ``cdp_response``.

    Bypasses ``__init__`` entirely so we don't need a real WS connection.  We
    set just the state ``evaluate_runtime`` reads.
    """
    import asyncio
    import threading

    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._active = True
    sup._page_session_id = "test-session-id"

    # Build a real running event loop on a background thread so
    # asyncio.run_coroutine_threadsafe has somewhere to dispatch.
    loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        return cdp_response

    sup._cdp = _fake_cdp  # type: ignore[method-assign]
    sup._loop = loop
    sup._thread = thread
    return sup


def _stop_supervisor(sup):
    sup._loop.call_soon_threadsafe(sup._loop.stop)
    sup._thread.join(timeout=2)


class TestEvaluateRuntimeResponseShaping:
    """CDPSupervisor.evaluate_runtime decodes the Runtime.evaluate response correctly."""

    def test_primitive_value(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {"result": {"type": "number", "value": 42}},
        })
        try:
            out = sup.evaluate_runtime("1 + 41")
            assert out == {"ok": True, "result": 42, "result_type": "number"}
        finally:
            _stop_supervisor(sup)

    def test_object_value_returned_by_value(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {
                "result": {
                    "type": "object",
                    "value": {"foo": "bar", "n": 7},
                }
            },
        })
        try:
            out = sup.evaluate_runtime('({foo:"bar", n:7})')
            assert out["ok"] is True
            assert out["result"] == {"foo": "bar", "n": 7}
            assert out["result_type"] == "object"
        finally:
            _stop_supervisor(sup)

    def test_undefined_value(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {"result": {"type": "undefined"}},
        })
        try:
            out = sup.evaluate_runtime("undefined")
            assert out == {"ok": True, "result": None, "result_type": "undefined"}
        finally:
            _stop_supervisor(sup)

    def test_dom_node_returns_description(self):
        """Non-serializable values (DOM nodes, functions) come back as description strings."""
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {
                "result": {
                    "type": "object",
                    "subtype": "node",
                    "description": "div#main.app",
                    # No 'value' key — returnByValue couldn't serialize it.
                }
            },
        })
        try:
            out = sup.evaluate_runtime("document.querySelector('#main')")
            assert out["ok"] is True
            assert out["result"] == "div#main.app"
            assert out["result_type"] == "object"
        finally:
            _stop_supervisor(sup)

    def test_js_exception_returns_error(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {
                "result": {"type": "undefined"},
                "exceptionDetails": {
                    "text": "Uncaught",
                    "exception": {
                        "description": "ReferenceError: foo is not defined",
                    },
                },
            },
        })
        try:
            out = sup.evaluate_runtime("foo.bar")
            assert out["ok"] is False
            assert "ReferenceError" in out["error"]
        finally:
            _stop_supervisor(sup)

    def test_inactive_supervisor_returns_error_without_dispatch(self):
        """Inactive supervisor short-circuits before even touching the loop."""
        import threading
        from tools.browser_supervisor import CDPSupervisor

        sup = object.__new__(CDPSupervisor)
        sup._state_lock = threading.Lock()
        sup._active = False  # ← key
        sup._page_session_id = None
        sup._loop = None

        out = sup.evaluate_runtime("1+1")
        assert out["ok"] is False
        # Either "loop is not running" or "is not active" is acceptable —
        # both are caught by the supervisor-side error branch in _browser_eval.
        assert "supervisor" in out["error"].lower()

    def test_no_session_attached_returns_error(self):
        import asyncio
        import threading
        from tools.browser_supervisor import CDPSupervisor

        sup = object.__new__(CDPSupervisor)
        sup._state_lock = threading.Lock()
        sup._active = True
        sup._page_session_id = None  # ← attach hasn't happened yet

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
            daemon=True,
        )
        thread.start()
        sup._loop = loop
        try:
            out = sup.evaluate_runtime("1+1")
            assert out["ok"] is False
            assert "session" in out["error"].lower()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
