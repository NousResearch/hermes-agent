"""Focused tests for CloakBrowser runtime timeout/error handling.

Verifies that _run_async, _ensure_session, and cloakbrowser_navigate
fail fast with a clear JSON error instead of hanging forever when
launch, navigation, or other async operations stall.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import browser_cloakbrowser as bc


@pytest.fixture(autouse=True)
def _reset_sessions():
    for task_id in list(bc._sessions):
        bc.cloakbrowser_close(task_id)
    yield
    for task_id in list(bc._sessions):
        bc.cloakbrowser_close(task_id)


async def _never_completes(*args, **kwargs):
    await asyncio.sleep(9999)


def _awaitable(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


def _wait_for_thread_stop(thread, timeout=2.0):
    deadline = time.time() + timeout
    while thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    assert not thread.is_alive()


class _LoopBoundPage:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.url = "about:blank"
        self.titles = []
        self.snapshots = []

    def _check_loop(self):
        current = asyncio.get_running_loop()
        if current is not self.loop:
            raise RuntimeError("page used from different event loop")

    async def goto(self, url, **kwargs):
        self._check_loop()
        self.url = url
        return None

    async def title(self):
        self._check_loop()
        self.titles.append(self.url)
        return "Loop Bound"

    async def evaluate(self, script, payload=None):
        self._check_loop()
        self.snapshots.append((script, payload))
        return {"snapshot": f"snapshot for {self.url}", "refs": {"e1": {"selector": "#ok"}}, "element_count": 1}


class _LoopBoundClosable:
    def __init__(self, name, events, fail=False):
        self.name = name
        self.events = events
        self.fail = fail
        self.loop = asyncio.get_running_loop()
        self.close_calls = 0

    async def close(self):
        current = asyncio.get_running_loop()
        if current is not self.loop:
            raise RuntimeError(f"{self.name} closed from different event loop")
        self.close_calls += 1
        self.events.append((self.name, "close"))
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")
        return None


class TestRunAsyncTimeout:
    """_run_async must surface a clear RuntimeError when a coroutine stalls."""

    def test_timeout_raises_clear_runtime_error(self):
        with pytest.raises(RuntimeError, match="timed out"):
            bc._run_async(_never_completes(), timeout=0.001)

    def test_fast_coro_completes_normally_with_timeout(self):
        async def fast():
            return 42

        assert bc._run_async(fast(), timeout=5) == 42

    def test_fast_coro_completes_normally_without_explicit_timeout(self):
        async def fast():
            return "ok"

        assert bc._run_async(fast()) == "ok"


class TestEnsureSessionTimeout:
    """_ensure_session must surface timeout when launch stalls."""

    def test_hanging_launch_returns_clear_error(self):
        with patch.object(bc, "_launch_session", _never_completes), \
             patch.object(bc, "_CLOAKBROWSER_LAUNCH_TIMEOUT", 0.05):
            with pytest.raises(RuntimeError, match="timed out"):
                bc._ensure_session("test-launch-hang")

    def test_hanging_operation_cancels_session_and_stops_loop_thread(self):
        async def fake_launch_session():
            page = _LoopBoundPage()
            return {"page": page, "context": SimpleNamespace(), "refs": {}}

        with patch.object(bc, "_launch_session", fake_launch_session):
            session = bc._ensure_session("timeout-poison")

        thread = session["_thread"]
        with patch.object(bc, "_CLOAKBROWSER_NAV_TIMEOUT", 0.05), \
             patch.object(bc, "_navigate_page", _never_completes):
            result = json.loads(bc.cloakbrowser_navigate("https://example.com/hang", task_id="timeout-poison"))

        assert result["success"] is False
        assert "timed out" in result["error"]
        assert "timeout-poison" not in bc._sessions
        _wait_for_thread_stop(thread)

    def test_session_bound_page_stays_on_launch_loop_for_navigate_and_snapshot(self):
        async def fake_launch_session():
            page = _LoopBoundPage()
            return {"page": page, "context": SimpleNamespace(), "refs": {}}

        with patch.object(bc, "_launch_session", fake_launch_session):
            navigate = json.loads(bc.cloakbrowser_navigate("https://example.com/loop", task_id="loop-bound"))
            snapshot = json.loads(bc.cloakbrowser_snapshot(task_id="loop-bound"))

        assert navigate == {
            "success": True,
            "url": "https://example.com/loop",
            "title": "Loop Bound",
            "snapshot": "snapshot for https://example.com/loop",
            "element_count": 1,
        }
        assert snapshot == {
            "success": True,
            "snapshot": "snapshot for https://example.com/loop",
            "element_count": 1,
        }
        session = bc._sessions["loop-bound"]
        assert session["page"].loop is session["_loop"]
        assert bc.cloakbrowser_close("loop-bound") is True
        _wait_for_thread_stop(session["_thread"])
        assert "loop-bound" not in bc._sessions

    def test_close_teardown_closes_persistent_page_and_context_before_thread_stop(self):
        events = []

        async def fake_launch_session():
            page = _LoopBoundClosable("page", events)
            context = _LoopBoundClosable("context", events)
            return {
                "page": page,
                "context": context,
                "browser": None,
                "refs": {},
                "persistent": True,
            }

        with patch.object(bc, "_launch_session", fake_launch_session):
            session = bc._ensure_session("persistent-close")

        assert bc.cloakbrowser_close("persistent-close") is True
        _wait_for_thread_stop(session["_thread"])
        assert events == [("page", "close"), ("context", "close")]
        assert session["page"].close_calls == 1
        assert session["context"].close_calls == 1
        assert "persistent-close" not in bc._sessions

    def test_teardown_best_effort_closes_browser_even_if_page_and_context_close_fail(self):
        events = []

        async def fake_launch_session():
            page = _LoopBoundClosable("page", events, fail=True)
            context = _LoopBoundClosable("context", events, fail=True)
            browser = _LoopBoundClosable("browser", events)
            return {
                "page": page,
                "context": context,
                "browser": browser,
                "refs": {},
                "persistent": False,
            }

        with patch.object(bc, "_launch_session", fake_launch_session):
            session = bc._ensure_session("best-effort-close")

        assert bc.cloakbrowser_close("best-effort-close") is True
        _wait_for_thread_stop(session["_thread"])
        assert events == [("page", "close"), ("context", "close"), ("browser", "close")]
        assert session["browser"].close_calls == 1
        assert "best-effort-close" not in bc._sessions

    def test_close_all_tears_down_every_live_session(self):
        events = []

        async def fake_launch_session():
            page = _LoopBoundClosable("page", events)
            context = _LoopBoundClosable("context", events)
            return {
                "page": page,
                "context": context,
                "browser": None,
                "refs": {},
                "persistent": True,
            }

        with patch.object(bc, "_launch_session", fake_launch_session):
            headed = bc._ensure_session("headed-session")
            headless = bc._ensure_session("headless-session")

        assert bc.cloakbrowser_close_all() == 2
        _wait_for_thread_stop(headed["_thread"])
        _wait_for_thread_stop(headless["_thread"])
        assert "headed-session" not in bc._sessions
        assert "headless-session" not in bc._sessions


class TestNavigateTimeout:
    """cloakbrowser_navigate must return JSON error instead of hanging."""

    def test_hanging_launch_returns_json_error(self):
        with patch.object(bc, "_launch_session", _never_completes), \
             patch.object(bc, "_CLOAKBROWSER_LAUNCH_TIMEOUT", 0.05):
            result = json.loads(bc.cloakbrowser_navigate("https://example.com", task_id="test-nav-launch-hang"))
            assert result["success"] is False
            assert "timed out" in result["error"]

    def test_hanging_navigate_returns_json_error(self, monkeypatch):
        page = SimpleNamespace(url="https://example.com", title=_awaitable("Test"))
        session = {"page": page, "refs": {}}
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_CLOAKBROWSER_NAV_TIMEOUT", 0.05)
        monkeypatch.setattr(bc, "_navigate_page", _never_completes)

        result = json.loads(bc.cloakbrowser_navigate("https://example.com/hang", task_id="test-nav-hang"))
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_hanging_title_returns_json_error(self, monkeypatch):
        page = SimpleNamespace(url="https://example.com", title=_never_completes)
        session = {"page": page, "refs": {}}
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_navigate_page", lambda page, url, timeout=None: _awaitable(None)())
        monkeypatch.setattr(bc, "_CLOAKBROWSER_DEFAULT_TIMEOUT", 0.05)

        result = json.loads(bc.cloakbrowser_navigate("https://example.com", task_id="test-title-hang"))
        assert result["success"] is False
        assert "timed out" in result["error"]


class TestNormalBehaviorPreserved:
    """Existing non-hanging paths must still work correctly."""

    def test_navigate_succeeds_with_mocked_session(self, monkeypatch):
        page = SimpleNamespace(url="https://example.com/final", title=_awaitable("Example Title"))
        session = {"page": page, "refs": {}}
        navigated = []

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(
            bc, "_navigate_page",
            lambda page, url, timeout=None: navigated.append((page, url)) or _awaitable(None)(),
        )
        monkeypatch.setattr(
            bc, "_snapshot_page",
            _awaitable({"snapshot": "- link \"Docs\" [@e1]", "element_count": 1}),
        )

        result = json.loads(bc.cloakbrowser_navigate("https://example.com/start", task_id="test-normal-nav"))
        assert result == {
            "success": True,
            "url": "https://example.com/final",
            "title": "Example Title",
            "snapshot": "- link \"Docs\" [@e1]",
            "element_count": 1,
        }
        assert navigated == [(page, "https://example.com/start")]

    def test_navigate_passes_timeout_to_page_goto(self, monkeypatch):
        page = SimpleNamespace(url="https://example.com", title=_awaitable("T"))
        session = {"page": page, "refs": {}}
        captured = {}

        async def tracking_navigate(p, url, timeout=None):
            captured["timeout"] = timeout
            return None

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_navigate_page", tracking_navigate)
        monkeypatch.setattr(bc, "_snapshot_page", _awaitable({"snapshot": "", "element_count": 0}))

        bc.cloakbrowser_navigate("https://example.com", task_id="test-timeout-prop")
        assert captured["timeout"] == bc._CLOAKBROWSER_NAV_TIMEOUT

    def test_click_type_scroll_back_press_succeed(self, monkeypatch):
        page = SimpleNamespace(
            url="https://example.com/after",
            title=_awaitable("After"),
            go_back=_awaitable(None),
        )
        page.keyboard = SimpleNamespace(press=_awaitable(None))
        page.mouse = SimpleNamespace(wheel=_awaitable(None))

        session = {
            "page": page,
            "refs": {
                "e1": {"selector": "#submit"},
                "e2": {"selector": "#name"},
            },
        }

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(
            bc, "_snapshot_page",
            _awaitable({
                "snapshot": "- button \"Submit\" [@e1]",
                "element_count": 2,
                "refs": {"e1": {"selector": "#submit"}, "e2": {"selector": "#name"}},
            }),
        )
        monkeypatch.setattr(
            bc, "_click_selector",
            _awaitable(None),
        )
        monkeypatch.setattr(
            bc, "_type_into_selector",
            _awaitable(None),
        )

        snap = json.loads(bc.cloakbrowser_snapshot(task_id="test-ops"))
        assert snap["success"] is True, f"snapshot failed: {snap}"

        click = json.loads(bc.cloakbrowser_click("@e1", task_id="test-ops"))
        assert click["success"] is True, f"click failed: {click}"

        typed = json.loads(bc.cloakbrowser_type("e2", "hello", task_id="test-ops"))
        assert typed["success"] is True, f"type failed: {typed}"

        scroll = json.loads(bc.cloakbrowser_scroll("down", task_id="test-ops"))
        assert scroll["success"] is True, f"scroll failed: {scroll}"

        back = json.loads(bc.cloakbrowser_back(task_id="test-ops"))
        assert back["success"] is True, f"back failed: {back}"

        press = json.loads(bc.cloakbrowser_press("Enter", task_id="test-ops"))
        assert press["success"] is True, f"press failed: {press}"


class TestCloakBrowserPersistencePolicy:
    def test_inactivity_cleanup_closes_native_cloakbrowser_session(self, monkeypatch):
        from tools import browser_tool as bt

        closed = []
        monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr(bt, "cloakbrowser_close", lambda task_id=None: closed.append(task_id) or True)
        monkeypatch.setattr(bt, "_active_sessions", {})
        monkeypatch.setattr(bt, "_last_active_session_key", {})
        monkeypatch.setattr(bt, "_session_last_activity", {"idle-task": 0.0})
        monkeypatch.setattr(bt, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 30)
        monkeypatch.setattr(bt.time, "time", lambda: 100.0)

        bt._cleanup_inactive_browser_sessions()

        assert closed == ["idle-task"]
        assert "idle-task" not in bt._session_last_activity

    def test_explicit_cleanup_closes_native_cloakbrowser_session(self, monkeypatch):
        from tools import browser_tool as bt

        closed = []
        monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr(bt, "cloakbrowser_close", lambda task_id=None: closed.append(task_id) or True)
        monkeypatch.setattr(bt, "_active_sessions", {})
        monkeypatch.setattr(bt, "_last_active_session_key", {})
        monkeypatch.setattr(bt, "_session_last_activity", {"cleanup-task": 50.0})

        bt.cleanup_browser("cleanup-task")

        assert closed == ["cleanup-task"]
        assert "cleanup-task" not in bt._session_last_activity
