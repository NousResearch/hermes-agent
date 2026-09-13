"""Regression tests for the browser_cdp read-only capability boundary."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
import websockets
from websockets.asyncio.server import serve

from tools import browser_cdp_tool
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_install as bt_install


class _CDPServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._port = 0

    def start(self) -> str:
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def handler(ws) -> None:
                async for raw in ws:
                    request = json.loads(raw)
                    self.requests.append(request)
                    await ws.send(json.dumps({"id": request["id"], "result": {"method": request["method"]}}))

            async def serve_forever() -> None:
                self._server = await serve(handler, "127.0.0.1", 0)
                self._port = next(iter(self._server.sockets)).getsockname()[1]
                ready.set()
                await self._server.wait_closed()

            self._loop.run_until_complete(serve_forever())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(5)
        return f"ws://127.0.0.1:{self._port}/devtools/browser/mock"

    def stop(self) -> None:
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread:
            self._thread.join(3)


@pytest.fixture
def cdp_server(monkeypatch):
    server = _CDPServer()
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", server.start)
    try:
        yield server
    finally:
        server.stop()


def test_missing_method_returns_error():
    result = json.loads(browser_cdp_tool.browser_cdp(method=""))
    assert "method" in result["error"].lower()


@pytest.mark.parametrize("method", ["Browser.getVersion", "Target.getTargets"])
def test_read_only_allowlist_reaches_transport(cdp_server, method):
    result = json.loads(browser_cdp_tool.browser_cdp(method=method))

    assert result == {"success": True, "method": method, "result": {"method": method}}
    assert [request["method"] for request in cdp_server.requests] == [method]
    assert cdp_server.requests[0]["params"] == {}


@pytest.mark.parametrize("method, params", [
    ("Runtime.evaluate", {"expression": "fetch('http://' + '169.254.169.254/latest/meta-data/')"}),
    ("Runtime.callFunctionOn", {"functionDeclaration": "() => fetch('http://' + '169.254.169.254/')"}),
    ("Page.navigate", {"url": "http://169.254.169.254/latest/meta-data/"}),
    ("Network.enable", {}),
    ("DOM.setAttributeValue", {"nodeId": 1, "name": "src", "value": "http://169.254.169.254/"}),
    ("Page.addScriptToEvaluateOnNewDocument", {"source": "fetch('http://' + '169.254.169.254/')"}),
    ("Target.createTarget", {"url": "https://example.test"}),
    ("NotARealDomain.readAnything", {}),
])
def test_unsafe_or_unknown_methods_reject_before_transport(monkeypatch, method, params):
    calls = []
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "ws://127.0.0.1:9222/devtools/browser/mock")
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = json.loads(browser_cdp_tool.browser_cdp(method=method, params=params))

    assert result["error"] == browser_cdp_tool.CDP_CAPABILITY_ERROR
    assert calls == []


@pytest.mark.parametrize("field, value", [("params", {"unexpected": True}), ("target_id", "target-1"), ("frame_id", "frame-1")])
def test_allowlisted_methods_reject_arguments_that_expand_capability(monkeypatch, field, value):
    calls = []
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "ws://127.0.0.1:9222/devtools/browser/mock")
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = json.loads(browser_cdp_tool.browser_cdp(method="Target.getTargets", **{field: value}))

    assert result["error"] == browser_cdp_tool.CDP_CAPABILITY_ERROR
    assert calls == []


def test_rejection_happens_before_supervisor(monkeypatch):
    import tools.browser_supervisor as browser_supervisor

    monkeypatch.setattr(
        browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda *_args: pytest.fail("supervisor must not be consulted"),
    )

    result = json.loads(browser_cdp_tool.browser_cdp(
        method="Runtime.evaluate", params={"expression": "fetch('http://' + '169.254.169.254/')"}, frame_id="frame-1",
    ))

    assert result["error"] == browser_cdp_tool.CDP_CAPABILITY_ERROR


def test_no_endpoint_returns_helpful_error(monkeypatch):
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
    result = json.loads(browser_cdp_tool.browser_cdp(method="Target.getTargets"))
    assert "/browser connect" in result["error"]


def test_check_fn_does_not_probe_network(monkeypatch):
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    monkeypatch.setattr(bt_cdp, "_get_cdp_override_raw", lambda: "ws://127.0.0.1:9222/devtools/browser/x")
    assert browser_cdp_tool._browser_cdp_check() is True
