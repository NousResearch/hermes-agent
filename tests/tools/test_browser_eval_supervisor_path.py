"""Unit tests for the supervisor-WS fast path in browser_console / _browser_eval.

These exercise the dispatch logic in ``tools.browser_tool._browser_eval`` and
the response shaping in ``CDPSupervisor.evaluate_runtime`` using mocks — no
real browser, no real WebSocket.  Real-CDP coverage lives in
``tests/tools/test_browser_supervisor.py`` (gated on Chrome being installed).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
import threading
from unittest.mock import MagicMock

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


@pytest.fixture(autouse=True)
def _run_scheduled_coroutines_inline(monkeypatch):
    """Run mocked CDP coroutines deterministically without a background loop."""

    def schedule(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)


class _RunningLoop:
    def is_running(self):
        return True


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

    def test_closed_supervisor_session_falls_back_to_structured_tab_gone(self, monkeypatch):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": False,
            "kind": "cdp_protocol",
            "error": (
                "CDPProtocolError: CDP error on id=10: "
                "{'code': -32001, 'message': 'Session with given id not found.'}"
            ),
            "data": {
                "protocol_error": {
                    "code": -32001,
                    "message": "Session with given id not found.",
                },
                "session_lost": True,
            },
        }
        _patch_supervisor(monkeypatch, sup)
        stop = MagicMock()
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", stop)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: {
                "success": False,
                "error": "Pinned tab is no longer available",
                "code": "tab_gone",
                "data": {"targetId": "TARGET-A"},
            },
        )

        out = json.loads(bt._browser_eval("location.href"))

        assert out["success"] is False
        assert out["code"] == "tab_gone"
        assert out["data"] == {"targetId": "TARGET-A"}
        stop.assert_called_once_with("test-task")

    @pytest.mark.parametrize(
        "message",
        [
            "target closed",
            "session with given id not found",
        ],
    )
    def test_js_exception_text_never_replays_expression(self, monkeypatch, message):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.return_value = {
            "ok": False,
            "kind": "js_exception",
            "error": f"Uncaught Error: {message}",
            "data": {"exception_details": {"text": message}},
        }
        _patch_supervisor(monkeypatch, sup)
        stop = MagicMock()
        subprocess_run = MagicMock(side_effect=AssertionError("must not replay"))
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", stop)
        monkeypatch.setattr(bt, "_run_browser_command", subprocess_run)

        out = json.loads(bt._browser_eval("window.sideEffectCount++"))

        assert out["success"] is False
        assert out["code"] == "javascript_exception"
        assert message in out["error"]
        subprocess_run.assert_not_called()
        stop.assert_not_called()

    def test_ambiguous_supervisor_exception_is_not_replayed(self, monkeypatch):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime.side_effect = TimeoutError("response lost")
        _patch_supervisor(monkeypatch, sup)
        subprocess_run = MagicMock(side_effect=AssertionError("must not replay"))
        monkeypatch.setattr(bt, "_run_browser_command", subprocess_run)

        out = json.loads(bt._browser_eval("window.sideEffectCount++"))

        assert out["success"] is False
        assert out["code"] == "cdp_evaluate_failed"
        assert out["data"] == {"exception_type": "TimeoutError"}
        subprocess_run.assert_not_called()

    def test_subprocess_reference_chain_error_becomes_guidance(self, monkeypatch):
        """The CLI subprocess can't retry with returnByValue=False, so the
        cryptic 'Object reference chain is too long' CDP error must be turned
        into actionable guidance instead of surfaced raw."""
        import tools.browser_tool as bt

        # No supervisor → subprocess path runs.
        _patch_supervisor(monkeypatch, None)

        def _fake_subprocess(task_id, cmd, args):
            assert cmd == "eval"
            return {
                "success": False,
                "error": "Runtime.evaluate failed: Object reference chain is too long",
            }

        monkeypatch.setattr(bt, "_run_browser_command", _fake_subprocess)

        out = json.loads(bt._browser_eval("document.body"))
        assert out["success"] is False
        # Raw protocol error must NOT leak through.
        assert "reference chain" not in out["error"].lower()
        # Actionable guidance instead.
        assert "primitive" in out["error"].lower()
        assert "DOM node" in out["error"] or "dom node" in out["error"].lower()


# ---------------------------------------------------------------------------
# Response shaping: CDPSupervisor.evaluate_runtime
# ---------------------------------------------------------------------------


def _make_supervisor_with_cdp(cdp_response):
    """Build a CDPSupervisor instance that mocks ``_cdp`` to return ``cdp_response``.

    Bypasses ``__init__`` entirely so we don't need a real WS connection.  We
    set just the state ``evaluate_runtime`` reads.
    """
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._active = True
    sup._page_session_id = "test-session-id"

    async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        return cdp_response

    sup._cdp = _fake_cdp  # type: ignore[method-assign]
    sup._loop = _RunningLoop()
    return sup


def _stop_supervisor(sup):
    del sup


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


    def test_no_session_attached_returns_error(self):
        from tools.browser_supervisor import CDPSupervisor

        sup = object.__new__(CDPSupervisor)
        sup._state_lock = threading.Lock()
        sup._active = True
        sup._page_session_id = None  # ← attach hasn't happened yet

        sup._loop = _RunningLoop()
        out = sup.evaluate_runtime("1+1")
        assert out == {
            "ok": False,
            "kind": "supervisor_unavailable",
            "error": "supervisor has no attached page session",
            "data": {"reason": "no_page_session"},
        }

    def test_js_exception_details_are_structured(self):
        details = {
            "text": "Uncaught",
            "exception": {
                "description": "Error: session with given id not found"
            },
        }
        sup = _make_supervisor_with_cdp(
            {
                "id": 2,
                "result": {
                    "result": {"type": "object", "subtype": "error"},
                    "exceptionDetails": details,
                },
            }
        )

        out = sup.evaluate_runtime("throw new Error('session with given id not found')")

        assert out == {
            "ok": False,
            "kind": "js_exception",
            "error": "Uncaught: Error: session with given id not found",
            "data": {"exception_details": details},
        }

    def test_stale_protocol_error_is_structured(self):
        from tools.browser_supervisor import CDPProtocolError

        async def _stale_cdp(method, params=None, *, session_id=None, timeout=10.0):
            raise CDPProtocolError(
                9,
                {"code": -32001, "message": "Session with given id not found."},
            )

        sup = _make_supervisor_with_cdp_fn(_stale_cdp)
        out = sup.evaluate_runtime("location.href")

        assert out["ok"] is False
        assert out["kind"] == "cdp_protocol"
        assert out["data"] == {
            "protocol_error": {
                "code": -32001,
                "message": "Session with given id not found.",
            },
            "session_lost": True,
        }


def _make_supervisor_with_cdp_fn(cdp_fn):
    """Like ``_make_supervisor_with_cdp`` but lets the test supply a coroutine
    function as ``_cdp`` so behaviour can vary by params (e.g. returnByValue).
    """
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._active = True
    sup._page_session_id = "test-session-id"

    sup._cdp = cdp_fn  # type: ignore[method-assign]
    sup._loop = _RunningLoop()
    return sup


class TestEvaluateRuntimeDomNodeCrashRetry:
    """returnByValue=True on a DOM node fails CDP serialization with 'Object
    reference chain is too long'.  evaluate_runtime must retry with
    returnByValue=False and return the node's description instead of crashing.
    """

    def test_reference_chain_crash_retries_without_by_value(self):
        from tools.browser_supervisor import CDPProtocolError

        calls = []

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            by_value = (params or {}).get("returnByValue")
            calls.append(by_value)
            if by_value:
                raise CDPProtocolError(
                    7,
                    {
                        "code": -32000,
                        "message": "Object reference chain is too long",
                    },
                )
            # returnByValue=False: Chrome returns the node's description, no value.
            return {
                "id": 8,
                "result": {
                    "result": {
                        "type": "object",
                        "subtype": "node",
                        "description": "body",
                    }
                },
            }

        sup = _make_supervisor_with_cdp_fn(_fake_cdp)
        try:
            out = sup.evaluate_runtime("document.body")
            assert out["ok"] is True
            assert out["result"] == "body"
            assert out["result_type"] == "object"
            # First call by_value=True (crashed), retried with by_value=False.
            assert calls == [True, False]
        finally:
            _stop_supervisor(sup)

    def test_unrelated_error_does_not_retry(self):
        calls = []

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            calls.append((params or {}).get("returnByValue"))
            raise RuntimeError("CDP error on id=3: {'message': 'Target closed'}")

        sup = _make_supervisor_with_cdp_fn(_fake_cdp)
        try:
            out = sup.evaluate_runtime("document.body")
            assert out["ok"] is False
            assert out["kind"] == "cdp_transport"
            assert "Target closed" in out["error"]
            # No retry for unrelated failures — exactly one call.
            assert calls == [True]
        finally:
            _stop_supervisor(sup)
