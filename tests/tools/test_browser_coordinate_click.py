"""Tests for compositor-level coordinate click (browser_click with x/y params).

Covers:
- Input validation (ref vs x/y mutually exclusive)
- CDP coordinate click path (via mock CDP server)
- agent-browser mouse fallback path
- Camofox passthrough still works with ref
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List
import pytest

import websockets
from websockets.asyncio.server import serve


# ---------------------------------------------------------------------------
# In-process CDP mock server (reused from test_browser_cdp_tool.py)
# ---------------------------------------------------------------------------


class _CDPServer:
    """Tiny CDP mock — replies to registered method handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Any] = {}
        self._responses: List[Dict[str, Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._host = "127.0.0.1"
        self._port = 0
        self._url: str = ""

    def on(self, method: str, handler):
        self._handlers[method] = handler

    def start(self) -> str:
        ready = threading.Event()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _handler(ws):
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        call_id = msg.get("id")
                        method = msg.get("method", "")
                        params = msg.get("params", {}) or {}
                        session_id = msg.get("sessionId")
                        self._responses.append(msg)

                        fn = self._handlers.get(method)
                        if fn is None:
                            reply = {
                                "id": call_id,
                                "error": {"code": -32601, "message": f"No handler for {method}"},
                            }
                        else:
                            try:
                                result = fn(params, session_id)
                                reply = {"id": call_id, "result": result}
                            except Exception as exc:
                                reply = {"id": call_id, "error": {"code": -1, "message": str(exc)}}
                        if session_id:
                            reply["sessionId"] = session_id
                        await ws.send(json.dumps(reply))
                except websockets.exceptions.ConnectionClosed:
                    pass

            async def _serve() -> None:
                self._server = await serve(_handler, self._host, 0)
                sock = next(iter(self._server.sockets))
                self._port = sock.getsockname()[1]
                ready.set()
                await self._server.wait_closed()

            try:
                self._loop.run_until_complete(_serve())
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("CDP mock server failed to start")
        self._url = f"ws://{self._host}:{self._port}/devtools/browser/mock"
        return self._url

    def stop(self) -> None:
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread:
            self._thread.join(timeout=3.0)

    def received(self) -> List[Dict[str, Any]]:
        return list(self._responses)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cdp_server(monkeypatch):
    """Start a CDP mock and point browser_cdp_tool's resolver at it."""
    server = _CDPServer()
    ws_url = server.start()

    import tools.browser_cdp_tool as cdp_mod
    monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: ws_url)

    # clear the session cache so each test starts fresh
    from tools import browser_tool as _bt
    _bt._CDP_SESSION_CACHE.clear()

    try:
        yield server
    finally:
        _bt._CDP_SESSION_CACHE.clear()
        server.stop()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestClickInputValidation:
    """browser_click validates that exactly one of ref / (x,y) is provided."""

    def test_neither_ref_nor_coords(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click())
        assert result["success"] is False
        assert "ref" in result["error"].lower() or "x" in result["error"].lower()

    def test_both_ref_and_coords(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(ref="@e1", x=100, y=200))
        assert result["success"] is False
        assert "not both" in result["error"].lower()

    def test_x_without_y(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(x=100))
        assert result["success"] is False
        assert "both" in result["error"].lower()

    def test_y_without_x(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(y=200))
        assert result["success"] is False
        assert "both" in result["error"].lower()

    def test_empty_ref_treated_as_missing(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(ref=""))
        assert result["success"] is False
        assert "ref" in result["error"].lower() or "x" in result["error"].lower()

    def test_non_numeric_coordinates(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(x="abc", y="def"))
        assert result["success"] is False
        assert "number" in result["error"].lower()


# ---------------------------------------------------------------------------
# CDP coordinate click (happy path via mock server)
# ---------------------------------------------------------------------------


class TestCDPCoordinateClick:
    """Coordinate clicks via CDP Input.dispatchMouseEvent."""

    def test_cdp_click_dispatches_press_and_release(self, cdp_server):
        from tools.browser_tool import browser_click

        # Register handlers for the protocol calls
        cdp_server.on(
            "Target.getTargets",
            lambda p, s: {
                "targetInfos": [
                    {"targetId": "page-1", "type": "page", "attached": True, "url": "https://example.com"},
                ]
            },
        )
        cdp_server.on(
            "Target.attachToTarget",
            lambda p, s: {"sessionId": f"sess-{p['targetId']}"},
        )
        cdp_server.on(
            "Input.dispatchMouseEvent",
            lambda p, s: {},
        )

        result = json.loads(browser_click(x=150, y=300))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 150, "y": 300}
        assert result["method"] == "cdp_compositor"

        # Verify the CDP calls: Target.getTargets, attach, mousePressed, attach, mouseReleased
        calls = cdp_server.received()
        methods = [c["method"] for c in calls]
        assert "Target.getTargets" in methods
        assert "Input.dispatchMouseEvent" in methods

        # Find the mouse events
        mouse_events = [c for c in calls if c["method"] == "Input.dispatchMouseEvent"]
        assert len(mouse_events) == 2
        assert mouse_events[0]["params"]["type"] == "mousePressed"
        assert mouse_events[0]["params"]["x"] == 150
        assert mouse_events[0]["params"]["y"] == 300
        assert mouse_events[0]["params"]["button"] == "left"
        assert mouse_events[1]["params"]["type"] == "mouseReleased"

    def test_cdp_click_rounds_float_coordinates(self, cdp_server):
        from tools.browser_tool import browser_click

        cdp_server.on(
            "Target.getTargets",
            lambda p, s: {"targetInfos": [{"targetId": "p1", "type": "page", "attached": True, "url": "..."}]},
        )
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": "s1"})
        cdp_server.on("Input.dispatchMouseEvent", lambda p, s: {})

        result = json.loads(browser_click(x=150.7, y=299.3))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 151, "y": 299}

    def test_cdp_click_no_page_target_still_works(self, cdp_server):
        """When Target.getTargets returns no page targets, click proceeds without target_id."""
        from tools.browser_tool import browser_click

        cdp_server.on(
            "Target.getTargets",
            lambda p, s: {"targetInfos": [{"targetId": "sw1", "type": "service_worker"}]},
        )
        # No Target.attachToTarget needed — page_target is None so _cdp_call
        # sends without attaching
        cdp_server.on("Input.dispatchMouseEvent", lambda p, s: {})

        result = json.loads(browser_click(x=50, y=50))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 50, "y": 50}

    def test_cdp_dispatch_mouse_event_failure(self, cdp_server):
        """When Input.dispatchMouseEvent returns a CDP error, return failure."""
        from tools.browser_tool import browser_click

        cdp_server.on(
            "Target.getTargets",
            lambda p, s: {"targetInfos": [{"targetId": "p1", "type": "page", "attached": True, "url": "..."}]},
        )
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": "s1"})
        # No handler for Input.dispatchMouseEvent — server returns CDP error

        result = json.loads(browser_click(x=100, y=200))
        assert result["success"] is False
        assert "CDP coordinate click failed" in result["error"]


# ---------------------------------------------------------------------------
# agent-browser mouse fallback
# ---------------------------------------------------------------------------


class TestAgentBrowserMouseFallback:
    """When no CDP endpoint is available, fall back to agent-browser mouse commands."""

    def test_falls_back_to_agent_browser_mouse(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        # No CDP endpoint available
        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")

        # Mock _run_browser_command and _last_session_key
        commands_sent = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            commands_sent.append((command, args))
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        result = json.loads(browser_tool.browser_click(x=200, y=400))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 200, "y": 400}
        assert result["method"] == "agent_browser_mouse"

        # Should have sent: mouse move, mouse down, mouse up
        assert len(commands_sent) == 3
        assert commands_sent[0] == ("mouse", ["move", "200", "400"])
        assert commands_sent[1] == ("mouse", ["down"])
        assert commands_sent[2] == ("mouse", ["up"])

    def test_mouse_move_failure_returns_error(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if args and args[0] == "move":
                return {"success": False, "error": "mouse move not supported"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        result = json.loads(browser_tool.browser_click(x=100, y=100))
        assert result["success"] is False
        assert "mouse move" in result["error"]

    def test_mouse_down_failure_returns_error(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if args and args[0] == "down":
                return {"success": False, "error": "mouse down failed"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        result = json.loads(browser_tool.browser_click(x=100, y=100))
        assert result["success"] is False
        assert "mouse down" in result["error"]

    def test_mouse_up_failure_returns_error(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if args and args[0] == "up":
                return {"success": False, "error": "mouse up failed"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        result = json.loads(browser_tool.browser_click(x=100, y=100))
        assert result["success"] is False
        assert "mouse up" in result["error"]


# ---------------------------------------------------------------------------
# Ref-based click unchanged
# ---------------------------------------------------------------------------


class TestRefClickPreserved:
    """Existing ref-based click behavior is unchanged."""

    def test_ref_click_still_works(self, monkeypatch):
        from tools import browser_tool

        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e5"))
        assert result["success"] is True
        assert result["clicked"] == "@e5"

    def test_ref_without_at_prefix_auto_added(self, monkeypatch):
        from tools import browser_tool

        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        clicked_refs = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            clicked_refs.append(args)
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        browser_tool.browser_click(ref="e12")
        assert clicked_refs[0] == ["@e12"]


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------


class TestSchemaUpdated:
    """The tool schema reflects x/y params and ref is no longer required."""

    def test_schema_has_x_y_properties(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP["browser_click"]
        props = schema["parameters"]["properties"]
        assert "x" in props
        assert "y" in props
        assert props["x"]["type"] == "number"
        assert props["y"]["type"] == "number"

    def test_schema_no_required_fields(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP["browser_click"]
        # ref is no longer required — either ref or x+y
        assert "required" not in schema["parameters"] or schema["parameters"]["required"] == []

    def test_schema_ref_still_present(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP["browser_click"]
        assert "ref" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """browser_click is registered with x/y params wired through."""

    def test_dispatch_with_coordinates(self, monkeypatch, cdp_server):
        from tools.registry import registry

        cdp_server.on(
            "Target.getTargets",
            lambda p, s: {"targetInfos": [{"targetId": "p1", "type": "page", "attached": True, "url": "..."}]},
        )
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": "s1"})
        cdp_server.on("Input.dispatchMouseEvent", lambda p, s: {})

        raw = registry.dispatch(
            "browser_click", {"x": 42, "y": 84}, task_id="t1"
        )
        result = json.loads(raw)
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 42, "y": 84}

    def test_dispatch_with_ref(self, monkeypatch):
        from tools import browser_tool
        from tools.registry import registry

        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)
        monkeypatch.setattr(
            browser_tool, "_run_browser_command",
            lambda tid, cmd, args=None, timeout=None: {"success": True},
        )

        raw = registry.dispatch("browser_click", {"ref": "@e3"}, task_id="t1")
        result = json.loads(raw)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Session caching
# ---------------------------------------------------------------------------


class TestSessionCaching:
    """Second click skips Target.getTargets + Target.attachToTarget."""

    def test_second_click_skips_session_resolution(self, cdp_server, monkeypatch):
        """After first click the session_id is cached; second click goes straight
        to mousePressed+mouseReleased without re-issuing getTargets/attachToTarget."""
        from tools import browser_tool
        import tools.browser_cdp_tool as cdp_mod

        # clear cache
        browser_tool._CDP_SESSION_CACHE.clear()
        monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: cdp_server._url)

        resolve_count = {"n": 0}

        def _getTargets(p, s):
            resolve_count["n"] += 1
            return {"targetInfos": [{"targetId": "p1", "type": "page", "attached": True, "url": "..."}]}

        cdp_server.on("Target.getTargets", _getTargets)
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": "sess-cached"})
        cdp_server.on("Input.dispatchMouseEvent", lambda p, s: {})

        # First click — must call getTargets
        r1 = json.loads(browser_tool.browser_click(x=10.0, y=20.0))
        assert r1["success"] is True
        assert resolve_count["n"] == 1

        # Second click — cache hit; getTargets must NOT be called again
        r2 = json.loads(browser_tool.browser_click(x=30.0, y=40.0))
        assert r2["success"] is True
        assert resolve_count["n"] == 1, "session resolution was repeated despite warm cache"

    def test_stale_session_triggers_reattach(self, cdp_server, monkeypatch):
        """If the browser returns 'Session with given id not found', the cache is
        cleared and session resolution runs again before retrying the click."""
        from tools import browser_tool
        import tools.browser_cdp_tool as cdp_mod

        browser_tool._CDP_SESSION_CACHE.clear()
        monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: cdp_server._url)

        call_count = {"mouse": 0, "resolve": 0}

        def _getTargets(p, s):
            call_count["resolve"] += 1
            return {"targetInfos": [{"targetId": "px", "type": "page", "attached": True, "url": "..."}]}

        def _dispatch(p, s):
            call_count["mouse"] += 1
            # First two mouse calls (with stale session) return an error;
            # after re-resolve they should succeed
            if call_count["mouse"] <= 2:
                raise RuntimeError("Session with given id not found: stale-session-id")
            return {}

        cdp_server.on("Target.getTargets", _getTargets)
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": f"sess-{call_count['resolve']}"})
        cdp_server.on("Input.dispatchMouseEvent", _dispatch)

        # Seed cache with stale session to trigger the error path
        browser_tool._CDP_SESSION_CACHE[(cdp_server._url, "default")] = "stale-session-id"

        r = json.loads(browser_tool.browser_click(x=50.0, y=60.0))
        assert r["success"] is True
        # Must have resolved the session once (after evicting stale entry)
        assert call_count["resolve"] >= 1

    def test_cache_cleared_on_endpoint_change(self, monkeypatch):
        """Cache is keyed per (endpoint URL, task_id); a different task_id on the
        same endpoint does not reuse another task's cached session."""
        from tools import browser_tool

        browser_tool._CDP_SESSION_CACHE.clear()
        browser_tool._CDP_SESSION_CACHE[("ws://endpoint-a/", "task-a")] = "sess-a"

        # A different task_id on the same endpoint must not find task-a's session
        assert browser_tool._CDP_SESSION_CACHE.get(("ws://endpoint-a/", "task-b")) is None

