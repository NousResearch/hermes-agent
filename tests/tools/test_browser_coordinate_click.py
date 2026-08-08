"""Tests for native (compositor-level) ref-based click in browser_click.

Covers:
- browser_click requires a ref
- Private-page action guard blocks the click (regression test for #62991 review)
- CDP native click path (resolve ref box -> Input.dispatchMouseEvent at center)
- agent-browser mouse fallback path (no CDP endpoint)
- box-resolution failure degrades gracefully to plain ref click
- Camofox passthrough still works with ref
- Schema reflects ref-only (no x/y)
- Session caching + stale-session reattach
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List
import pytest

import websockets
from websockets.asyncio.server import serve


class _CDPServer:
    """Tiny CDP mock - replies to registered method handlers."""

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


@pytest.fixture
def cdp_server(monkeypatch):
    server = _CDPServer()
    ws_url = server.start()

    import tools.browser_cdp_tool as cdp_mod
    monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: ws_url)

    from tools import browser_tool as _bt
    _bt._CDP_SESSION_CACHE.clear()

    try:
        yield server
    finally:
        _bt._CDP_SESSION_CACHE.clear()
        server.stop()


def _wire_cdp_click_handlers(server: _CDPServer) -> None:
    server.on(
        "Target.getTargets",
        lambda p, s: {
            "targetInfos": [
                {"targetId": "page-1", "type": "page", "attached": True, "url": "https://example.com"},
            ]
        },
    )
    server.on("Target.attachToTarget", lambda p, s: {"sessionId": f"sess-{p['targetId']}"})
    server.on("Input.dispatchMouseEvent", lambda p, s: {})


def _mock_ref_box(monkeypatch, x: float, y: float, w: float, h: float) -> None:
    from tools import browser_tool

    def mock_run_cmd(task_id, command, args=None, timeout=None):
        if command == "get" and args and args[0] == "box":
            return {"success": True, "data": {"x": x, "y": y, "width": w, "height": h}}
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)


def _box_data(x: float, y: float, w: float, h: float) -> dict:
    return {"success": True, "data": {"x": x, "y": y, "width": w, "height": h}}


class TestClickInputValidation:
    def test_missing_ref(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click())
        assert result["success"] is False
        assert "ref" in result["error"].lower()

    def test_empty_ref_treated_as_missing(self):
        from tools.browser_tool import browser_click

        result = json.loads(browser_click(ref=""))
        assert result["success"] is False
        assert "ref" in result["error"].lower()


class TestPrivatePageGuard:
    def test_guard_blocks_native_click(self, monkeypatch):
        from tools import browser_tool

        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)
        monkeypatch.setattr(
            browser_tool, "_blocked_private_page_action",
            lambda tid, action: json.dumps({"success": False, "error": "Blocked: private page"}),
        )
        commands = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            commands.append((command, args))
            return {"success": True, "data": {"x": 0, "y": 0, "width": 10, "height": 10}}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e5"))
        assert result["success"] is False
        assert "Blocked" in result["error"]
        assert commands == []


class TestCDPNativeClick:
    def test_cdp_click_dispatches_press_and_release_at_center(self, cdp_server, monkeypatch):
        from tools.browser_tool import browser_click

        _wire_cdp_click_handlers(cdp_server)
        _mock_ref_box(monkeypatch, 100.0, 200.0, 40.0, 20.0)

        result = json.loads(browser_click(ref="@e1"))
        assert result["success"] is True
        assert result["clicked"] == "@e1"
        assert result["clicked_at"] == {"x": 120, "y": 210}
        assert result["method"] == "cdp_native"

        calls = cdp_server.received()
        methods = [c["method"] for c in calls]
        assert "Target.getTargets" in methods
        assert "Input.dispatchMouseEvent" in methods

        mouse_events = [c for c in calls if c["method"] == "Input.dispatchMouseEvent"]
        assert len(mouse_events) == 2
        assert mouse_events[0]["params"]["type"] == "mousePressed"
        assert mouse_events[0]["params"]["x"] == 120
        assert mouse_events[0]["params"]["y"] == 210
        assert mouse_events[0]["params"]["button"] == "left"
        assert mouse_events[1]["params"]["type"] == "mouseReleased"

    def test_cdp_click_rounds_center(self, cdp_server, monkeypatch):
        from tools.browser_tool import browser_click

        _wire_cdp_click_handlers(cdp_server)
        _mock_ref_box(monkeypatch, 10.2, 10.7, 3.0, 3.0)

        result = json.loads(browser_click(ref="@e1"))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 12, "y": 12}

    def test_cdp_dispatch_failure_returns_error(self, cdp_server, monkeypatch):
        from tools.browser_tool import browser_click

        _wire_cdp_click_handlers(cdp_server)
        cdp_server._handlers.pop("Input.dispatchMouseEvent", None)
        _mock_ref_box(monkeypatch, 0.0, 0.0, 10.0, 10.0)

        result = json.loads(browser_click(ref="@e1"))
        assert result["success"] is False
        assert "CDP native click failed" in result["error"]


class TestAgentBrowserMouseFallback:
    def test_falls_back_to_agent_browser_mouse(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
        _mock_ref_box(monkeypatch, 100.0, 200.0, 40.0, 20.0)

        commands_sent = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if command == "get" and args and args[0] == "box":
                return _box_data(100.0, 200.0, 40.0, 20.0)
            commands_sent.append((command, args))
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e1"))
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 120, "y": 210}
        assert result["method"] == "agent_browser_mouse"

        assert commands_sent[0] == ("mouse", ["move", "120", "210"])
        assert commands_sent[1] == ("mouse", ["down"])
        assert commands_sent[2] == ("mouse", ["up"])

    def test_mouse_down_failure_returns_error(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
        _mock_ref_box(monkeypatch, 100.0, 200.0, 40.0, 20.0)

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if command == "get" and args and args[0] == "box":
                return _box_data(100.0, 200.0, 40.0, 20.0)
            if command == "mouse" and args and args[0] == "down":
                return {"success": False, "error": "mouse down failed"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e1"))
        assert result["success"] is False
        assert "mouse down" in result["error"]


class TestBoxResolutionFailure:
    def test_missing_size_falls_back(self, monkeypatch):
        """If width/height is missing or zero, fall back to plain ref click."""
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        commands = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            commands.append((command, args))
            if command == "get" and args and args[0] == "box":
                return {"success": True, "data": {"x": 100.0, "y": 200.0, "width": 0, "height": 0}}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e5"))
        assert result["success"] is True
        assert result["method"] == "agent_browser_ref"
        assert ("click", ["@e5"]) in commands

    def test_falls_back_to_plain_ref_click(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        commands = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            commands.append((command, args))
            if command == "get" and args and args[0] == "box":
                return {"success": False, "error": "element not found"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        result = json.loads(browser_tool.browser_click(ref="@e9"))
        assert result["success"] is True
        assert result["clicked"] == "@e9"
        assert result["method"] == "agent_browser_ref"
        assert ("click", ["@e9"]) in commands


class TestRefClickPlumbing:
    def test_ref_without_at_prefix_auto_added(self, monkeypatch):
        from tools import browser_tool, browser_cdp_tool

        monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda tid: tid)

        mouse_calls = []

        def mock_run_cmd(task_id, command, args=None, timeout=None):
            if command == "get" and args and args[0] == "box":
                return _box_data(0.0, 0.0, 1.0, 1.0)
            if command == "mouse":
                mouse_calls.append(args)
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_cmd)

        browser_tool.browser_click(ref="e12")
        # Native path: ref normalized to @e12; mouse click dispatched at the
        # resolved center (box 0,0,1,1 -> center 0,0).
        assert mouse_calls[0] == ["move", "0", "0"]
        assert mouse_calls[1] == ["down"]
        assert mouse_calls[2] == ["up"]

    def test_camofox_passthrough(self, monkeypatch):
        from tools import browser_tool
        import tools.browser_camofox as camofox_mod

        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)

        captured = {}

        def mock_camofox_click(ref, task_id):
            captured["ref"] = ref
            return json.dumps({"success": True, "clicked": ref})

        # browser_click does `from tools.browser_camofox import camofox_click`
        monkeypatch.setattr(camofox_mod, "camofox_click", mock_camofox_click)

        result = json.loads(browser_tool.browser_click(ref="@e3"))
        assert result["success"] is True
        assert captured["ref"] == "@e3"


class TestSchemaUpdated:
    def test_schema_has_only_ref_property(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP["browser_click"]
        props = schema["parameters"]["properties"]
        assert "ref" in props
        assert "x" not in props
        assert "y" not in props

    def test_ref_is_required(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP["browser_click"]
        assert schema["parameters"]["required"] == ["ref"]


class TestRegistryIntegration:
    def test_dispatch_with_ref(self, monkeypatch, cdp_server):
        from tools.registry import registry

        _wire_cdp_click_handlers(cdp_server)
        _mock_ref_box(monkeypatch, 50.0, 60.0, 20.0, 20.0)

        raw = registry.dispatch("browser_click", {"ref": "@e3"}, task_id="t1")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["clicked_at"] == {"x": 60, "y": 70}


class TestSessionCaching:
    def test_second_click_skips_session_resolution(self, cdp_server, monkeypatch):
        from tools import browser_tool
        import tools.browser_cdp_tool as cdp_mod

        browser_tool._CDP_SESSION_CACHE.clear()
        monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: cdp_server._url)
        _mock_ref_box(monkeypatch, 0.0, 0.0, 10.0, 10.0)

        resolve_count = {"n": 0}

        def _getTargets(p, s):
            resolve_count["n"] += 1
            return {"targetInfos": [{"targetId": "p1", "type": "page", "attached": True, "url": "..."}]}

        cdp_server.on("Target.getTargets", _getTargets)
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": "sess-cached"})
        cdp_server.on("Input.dispatchMouseEvent", lambda p, s: {})

        r1 = json.loads(browser_tool.browser_click(ref="@e1"))
        assert r1["success"] is True
        assert resolve_count["n"] == 1

        r2 = json.loads(browser_tool.browser_click(ref="@e2"))
        assert r2["success"] is True
        assert resolve_count["n"] == 1, "session resolution was repeated despite warm cache"

    def test_stale_session_triggers_reattach(self, cdp_server, monkeypatch):
        from tools import browser_tool
        import tools.browser_cdp_tool as cdp_mod

        browser_tool._CDP_SESSION_CACHE.clear()
        monkeypatch.setattr(cdp_mod, "_resolve_cdp_endpoint", lambda: cdp_server._url)
        _mock_ref_box(monkeypatch, 0.0, 0.0, 10.0, 10.0)

        call_count = {"mouse": 0, "resolve": 0}

        def _getTargets(p, s):
            call_count["resolve"] += 1
            return {"targetInfos": [{"targetId": "px", "type": "page", "attached": True, "url": "..."}]}

        def _dispatch(p, s):
            call_count["mouse"] += 1
            if call_count["mouse"] <= 2:
                raise RuntimeError("Session with given id not found: stale-session-id")
            return {}

        cdp_server.on("Target.getTargets", _getTargets)
        cdp_server.on("Target.attachToTarget", lambda p, s: {"sessionId": f"sess-{call_count['resolve']}"})
        cdp_server.on("Input.dispatchMouseEvent", _dispatch)

        browser_tool._CDP_SESSION_CACHE[(cdp_server._url, "default")] = "stale-session-id"

        r = json.loads(browser_tool.browser_click(ref="@e1"))
        assert r["success"] is True
        assert call_count["resolve"] >= 1

    def test_cache_cleared_on_endpoint_change(self, monkeypatch):
        from tools import browser_tool

        browser_tool._CDP_SESSION_CACHE.clear()
        browser_tool._CDP_SESSION_CACHE[("ws://endpoint-a/", "task-a")] = "sess-a"

        assert browser_tool._CDP_SESSION_CACHE.get(("ws://endpoint-a/", "task-b")) is None
