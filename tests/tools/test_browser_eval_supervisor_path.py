"""Unit tests for the supervisor-WS fast path in browser_console / _browser_eval.

These exercise the dispatch logic in ``tools.browser_tool._browser_eval`` and
the response shaping in ``CDPSupervisor.evaluate_runtime`` using mocks — no
real browser, no real WebSocket.  Real-CDP coverage lives in
``tests/tools/test_browser_supervisor.py`` (gated on Chrome being installed).
"""
from __future__ import annotations

import json
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


class TestSupervisorDaemonPageSplit:
    """The fast path is gated on the supervisor watching the daemon's page.

    The supervisor owns a separate CDP connection and attaches to the first
    page target *that connection* sees. On Browserless-style backends every
    CDP websocket gets a private browser, so the supervisor's connection can
    never see the page the agent-browser daemon navigated — its evals
    "succeed" against its own about:blank and return empty/wrong results
    (#32685 family). On plain Chrome the daemon may be driving a different
    tab. ``_supervisor_page_matches_daemon`` compares the supervisor's live
    top-frame URL with the last navigated URL and sends mismatches down the
    always-correct subprocess path.
    """

    @staticmethod
    def _sup_with_page(url, result="WRONG-BROWSER-RESULT"):
        sup = MagicMock()
        snap = MagicMock()
        snap.frame_tree = {
            "top": {"frame_id": "TOP", "url": url},
            "children": [],
        }
        sup.snapshot.return_value = snap
        sup.evaluate_runtime.return_value = {
            "ok": True,
            "result": result,
            "result_type": "string",
        }
        return sup

    def test_split_brain_supervisor_falls_through_to_subprocess(self, monkeypatch):
        """Supervisor stuck on about:blank while the daemon has a real page →
        the eval must come from the subprocess path, not the supervisor."""
        import tools.browser_tool as bt

        sup = self._sup_with_page("about:blank", result="")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(
            bt._last_navigated_urls, "test-task", "https://example.com/page"
        )
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: {"success": True, "data": {"result": "Example Domain"}},
        )

        out = json.loads(bt._browser_eval("document.title"))
        assert out["success"] is True
        assert out["result"] == "Example Domain"
        assert out.get("method") != "cdp_supervisor"
        sup.evaluate_runtime.assert_not_called()

    def test_matching_page_keeps_fast_path(self, monkeypatch):
        """Supervisor tracking the same URL the daemon navigated → fast path."""
        import tools.browser_tool as bt

        sup = self._sup_with_page("https://example.com/page", result="fast")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(
            bt._last_navigated_urls, "test-task", "https://example.com/page"
        )
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess must not run on a matching page"),
        )

        out = json.loads(bt._browser_eval("document.title"))
        assert out["success"] is True
        assert out["result"] == "fast"
        assert out["method"] == "cdp_supervisor"

    def test_fragment_only_difference_keeps_fast_path(self, monkeypatch):
        """#fragment navigation is the same document — fast path stays."""
        import tools.browser_tool as bt

        sup = self._sup_with_page("https://example.com/page#section-2", result="fast")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(
            bt._last_navigated_urls, "test-task", "https://example.com/page"
        )
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess must not run for fragment diffs"),
        )

        out = json.loads(bt._browser_eval("document.title"))
        assert out["method"] == "cdp_supervisor"

    def test_no_recorded_navigation_keeps_fast_path(self, monkeypatch):
        """No browser_navigate yet (e.g. /browser connect-then-eval) → nothing
        has diverged; legacy fast-path behaviour is preserved."""
        import tools.browser_tool as bt

        sup = self._sup_with_page("about:blank", result="legacy")
        _patch_supervisor(monkeypatch, sup)
        bt._last_navigated_urls.pop("test-task", None)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess must not run pre-navigation"),
        )

        out = json.loads(bt._browser_eval("1 + 1"))
        assert out["method"] == "cdp_supervisor"

    def test_unreadable_supervisor_state_falls_through(self, monkeypatch):
        """snapshot() blowing up must fail toward the subprocess path."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.snapshot.side_effect = RuntimeError("ws closed")
        sup.evaluate_runtime.return_value = {"ok": True, "result": "wrong"}
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(
            bt._last_navigated_urls, "test-task", "https://example.com/page"
        )
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *a, **kw: {"success": True, "data": {"result": "safe"}},
        )

        out = json.loads(bt._browser_eval("document.title"))
        assert out["result"] == "safe"
        assert out.get("method") != "cdp_supervisor"
        sup.evaluate_runtime.assert_not_called()


class TestPageDivergenceAfterInteractions:
    """The URL-match gate is only meaningful while the recorded URL still
    describes the daemon's page — review feedback on the gate's blind spot:

    A ``browser_click`` can open a new tab. The daemon rebinds to the new
    tab, the supervisor keeps showing the old page — whose URL still
    *matches* the recorded last-navigated URL — so the URL comparison alone
    would wrongly keep the fast path and eval in the old tab. Click results
    carry no URL (and probing the daemon would cost a subprocess call per
    click), so interactions that can move the page mark the session as
    possibly-diverged and the fast path stands down until the next
    authoritative URL: a navigate, or a back (whose result reports the
    landing URL) re-records and clears the mark.
    """

    URL = "https://example.com/page"

    @pytest.fixture(autouse=True)
    def _isolate_divergence_state(self, monkeypatch):
        import tools.browser_tool as bt

        # raising=False so a pre-fix run (no divergence tracking yet) shows
        # the behavioral failure instead of a fixture AttributeError.
        monkeypatch.setattr(bt, "_page_maybe_diverged", set(), raising=False)
        monkeypatch.setattr(bt, "_last_navigated_urls", {})
        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda *a, **kw: False)

    @staticmethod
    def _sup_with_page(url, result="WRONG-TAB-RESULT"):
        sup = MagicMock()
        snap = MagicMock()
        snap.frame_tree = {"top": {"frame_id": "TOP", "url": url}, "children": []}
        sup.snapshot.return_value = snap
        sup.evaluate_runtime.return_value = {
            "ok": True,
            "result": result,
            "result_type": "string",
        }
        return sup

    @staticmethod
    def _command_mux(responses):
        """_run_browser_command stand-in dispatching on the command name."""
        def _fake(task_id, cmd, args, **kwargs):
            assert cmd in responses, f"unexpected browser command: {cmd}"
            return responses[cmd]
        return _fake

    def test_click_then_eval_falls_through_to_subprocess(self, monkeypatch):
        """Reviewer scenario: click opened a new tab; supervisor still shows
        the old page whose URL matches the stale record. The eval must come
        from the subprocess path (daemon's page), not the supervisor's tab."""
        import tools.browser_tool as bt

        sup = self._sup_with_page(self.URL)
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "click": {"success": True},
            "eval": {"success": True, "data": {"result": "NEW-TAB-RESULT"}},
        }))

        assert json.loads(bt.browser_click("@e1"))["success"] is True
        out = json.loads(bt._browser_eval("document.title"))
        assert out["success"] is True
        assert out["result"] == "NEW-TAB-RESULT"
        assert out.get("method") != "cdp_supervisor"
        sup.evaluate_runtime.assert_not_called()

    def test_press_enter_then_eval_falls_through_to_subprocess(self, monkeypatch):
        """Enter can submit a form and navigate — same trust problem as a click."""
        import tools.browser_tool as bt

        sup = self._sup_with_page(self.URL)
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "press": {"success": True},
            "eval": {"success": True, "data": {"result": "POST-SUBMIT-RESULT"}},
        }))

        assert json.loads(bt.browser_press("Enter"))["success"] is True
        out = json.loads(bt._browser_eval("document.title"))
        assert out["result"] == "POST-SUBMIT-RESULT"
        assert out.get("method") != "cdp_supervisor"
        sup.evaluate_runtime.assert_not_called()

    def test_press_non_enter_keeps_fast_path(self, monkeypatch):
        """Typing/focus keys don't navigate — no reason to give up the fast path."""
        import tools.browser_tool as bt

        sup = self._sup_with_page(self.URL, result="fast")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "press": {"success": True},
        }))

        assert json.loads(bt.browser_press("Tab"))["success"] is True
        out = json.loads(bt._browser_eval("document.title"))
        assert out["result"] == "fast"
        assert out["method"] == "cdp_supervisor"

    def test_failed_click_keeps_fast_path(self, monkeypatch):
        """A click that never happened can't have moved the page."""
        import tools.browser_tool as bt

        sup = self._sup_with_page(self.URL, result="fast")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "click": {"success": False, "error": "no such ref"},
        }))

        assert json.loads(bt.browser_click("@e1"))["success"] is False
        out = json.loads(bt._browser_eval("document.title"))
        assert out["method"] == "cdp_supervisor"

    def test_back_reported_url_rerecords_and_restores_fast_path(self, monkeypatch):
        """back's result reports the landing URL — an authoritative daemon
        URL that both re-records the gate's comparison target and clears a
        prior click's divergence mark."""
        import tools.browser_tool as bt

        landing = "https://example.com/prev"
        sup = self._sup_with_page(landing, result="fast")
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "click": {"success": True},
            "back": {"success": True, "data": {"url": landing}},
        }))

        assert json.loads(bt.browser_click("@e1"))["success"] is True
        assert json.loads(bt.browser_back())["success"] is True
        out = json.loads(bt._browser_eval("document.title"))
        assert out["result"] == "fast"
        assert out["method"] == "cdp_supervisor"

    def test_back_without_reported_url_disables_fast_path(self, monkeypatch):
        """A back that can't say where it landed leaves the page unknown."""
        import tools.browser_tool as bt

        sup = self._sup_with_page(self.URL)
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setitem(bt._last_navigated_urls, "test-task", self.URL)
        monkeypatch.setattr(bt, "_run_browser_command", self._command_mux({
            "back": {"success": True, "data": {}},
            "eval": {"success": True, "data": {"result": "SUBPROCESS-RESULT"}},
        }))

        assert json.loads(bt.browser_back())["success"] is True
        out = json.loads(bt._browser_eval("document.title"))
        assert out["result"] == "SUBPROCESS-RESULT"
        sup.evaluate_runtime.assert_not_called()

    def test_record_daemon_url_clears_divergence(self, monkeypatch):
        """_record_daemon_url is what navigate/back call on success — it must
        clear the mark so a fresh navigation restores the fast path."""
        import tools.browser_tool as bt

        bt._page_maybe_diverged.add("test-task")
        sup = self._sup_with_page(self.URL)
        assert bt._supervisor_page_matches_daemon("test-task", sup) is False

        bt._record_daemon_url("test-task", self.URL)
        assert "test-task" not in bt._page_maybe_diverged
        assert bt._supervisor_page_matches_daemon("test-task", sup) is True


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


def _make_supervisor_with_cdp_fn(cdp_fn):
    """Like ``_make_supervisor_with_cdp`` but lets the test supply a coroutine
    function as ``_cdp`` so behaviour can vary by params (e.g. returnByValue).
    """
    import asyncio
    import threading

    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._state_lock = threading.Lock()
    sup._active = True
    sup._page_session_id = "test-session-id"

    loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    sup._cdp = cdp_fn  # type: ignore[method-assign]
    sup._loop = loop
    sup._thread = thread
    return sup


class TestEvaluateRuntimeDomNodeCrashRetry:
    """returnByValue=True on a DOM node fails CDP serialization with 'Object
    reference chain is too long'.  evaluate_runtime must retry with
    returnByValue=False and return the node's description instead of crashing.
    """

    def test_reference_chain_crash_retries_without_by_value(self):
        calls = []

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            by_value = (params or {}).get("returnByValue")
            calls.append(by_value)
            if by_value:
                # Mirror _read_loop turning a top-level CDP error into a RuntimeError.
                raise RuntimeError(
                    "CDP error on id=7: {'code': -32000, "
                    "'message': 'Object reference chain is too long'}"
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
            assert "Target closed" in out["error"]
            # No retry for unrelated failures — exactly one call.
            assert calls == [True]
        finally:
            _stop_supervisor(sup)
