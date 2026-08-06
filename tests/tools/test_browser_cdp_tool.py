"""Unit tests for browser_cdp tool.

Uses a tiny in-process ``websockets`` server to simulate a CDP endpoint —
gives real protocol coverage (connect, send, recv, close) without needing
a real Chrome instance.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List

import pytest

import websockets
from websockets.asyncio.server import serve

from tools import browser_cdp_tool


# ---------------------------------------------------------------------------
# In-process CDP mock server
# ---------------------------------------------------------------------------


class _CDPServer:
    """A tiny CDP-over-WebSocket mock.

    Each client gets a greeting-free stream.  The server replies to each
    inbound request whose ``id`` is set, using the registered handler for
    that method.  If no handler is registered, returns a generic CDP error.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Any] = {}
        self._responses: List[Dict[str, Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._host = "127.0.0.1"
        self._port = 0

    # --- handler registration --------------------------------------------

    def on(self, method: str, handler):
        """Register a handler ``handler(params, session_id) -> dict or Exception``."""
        self._handlers[method] = handler

    # --- lifecycle -------------------------------------------------------

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
                                "error": {
                                    "code": -32601,
                                    "message": f"No handler for {method}",
                                },
                            }
                        else:
                            try:
                                result = fn(params, session_id)
                                if isinstance(result, Exception):
                                    raise result
                                reply = {"id": call_id, "result": result}
                            except Exception as exc:
                                reply = {
                                    "id": call_id,
                                    "error": {"code": -1, "message": str(exc)},
                                }
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
            raise RuntimeError("CDP mock server failed to start within 5s")
        return f"ws://{self._host}:{self._port}/devtools/browser/mock"

    def stop(self) -> None:
        if self._loop and self._server:
            def _close() -> None:
                self._server.close()

            self._loop.call_soon_threadsafe(_close)
        if self._thread:
            self._thread.join(timeout=3.0)

    def received(self) -> List[Dict[str, Any]]:
        return list(self._responses)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cdp_server(monkeypatch):
    """Start a CDP mock and route tool resolution to it."""
    server = _CDPServer()
    ws_url = server.start()
    monkeypatch.setattr(
        browser_cdp_tool, "_resolve_cdp_endpoint", lambda: ws_url
    )
    try:
        yield server
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_method_returns_error():
    result = json.loads(browser_cdp_tool.browser_cdp(method=""))
    assert "error" in result
    assert "method" in result["error"].lower()
    assert result.get("cdp_docs") == browser_cdp_tool.CDP_DOCS_URL


def test_non_string_method_returns_error():
    result = json.loads(browser_cdp_tool.browser_cdp(method=123))  # type: ignore[arg-type]
    assert "error" in result
    assert "method" in result["error"].lower()


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_no_endpoint_returns_helpful_error(monkeypatch):
    monkeypatch.setattr(browser_cdp_tool, "_resolve_cdp_endpoint", lambda: "")
    result = json.loads(browser_cdp_tool.browser_cdp(method="Target.getTargets"))
    assert "error" in result
    assert "/browser connect" in result["error"]
    assert result.get("cdp_docs") == browser_cdp_tool.CDP_DOCS_URL


def test_websockets_missing_returns_error(monkeypatch):
    monkeypatch.setattr(browser_cdp_tool, "_WS_AVAILABLE", False)
    result = json.loads(browser_cdp_tool.browser_cdp(method="Target.getTargets"))
    assert "error" in result
    assert "websockets" in result["error"].lower()


# ---------------------------------------------------------------------------
# Happy-path: browser-level call
# ---------------------------------------------------------------------------


def test_browser_level_redacts_secret_result(cdp_server):
    fake_key = "sk-" + "CDPSECRETRESULT1234567890"
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": fake_key}},
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Runtime.evaluate"))

    assert result["success"] is True
    serialized = json.dumps(result)
    assert "CDPSECRETRESULT" not in serialized
    assert result["result"]["result"]["value"].startswith("sk-")


# ---------------------------------------------------------------------------
# Happy-path: target-attached call
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CDP error responses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timeout clamping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Private-network guard
# ---------------------------------------------------------------------------


PRIVATE_URL = "http://169.254.169.254/latest/meta-data/"


def test_runtime_evaluate_blocked_when_current_page_is_private(monkeypatch):
    calls = []

    monkeypatch.setattr(
        browser_cdp_tool,
        "_resolve_cdp_endpoint",
        lambda: "ws://127.0.0.1:9222/devtools/browser/mock",
    )

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: PRIVATE_URL)

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return {"result": {"value": "private data"}}

    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", fake_call)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert "private or internal address" in result["error"]
    assert calls == []


def test_frame_id_route_blocked_when_current_page_is_private(monkeypatch):
    """frame_id routing (OOPIF via supervisor) must not bypass the guard
    applied to the stateless path — same private-page boundary either way."""
    supervisor_calls = []

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: PRIVATE_URL)

    def fake_supervisor_route(**kwargs):
        supervisor_calls.append(kwargs)
        return json.dumps({"success": True, "result": {"value": "private data"}})

    monkeypatch.setattr(
        browser_cdp_tool, "_browser_cdp_via_supervisor", fake_supervisor_route
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="frame-1",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert "private or internal address" in result["error"]
    assert supervisor_calls == []


def test_frame_id_route_allowed_when_page_is_not_private(monkeypatch):
    """Sanity check: the new guard call must not block ordinary frame_id
    routing when the current page isn't private."""
    supervisor_calls = []

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    def fake_supervisor_route(**kwargs):
        supervisor_calls.append(kwargs)
        return json.dumps({"success": True, "result": {"value": "ok"}})

    monkeypatch.setattr(
        browser_cdp_tool, "_browser_cdp_via_supervisor", fake_supervisor_route
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            frame_id="frame-1",
            task_id="task-1",
        )
    )

    assert result.get("success") is True
    assert len(supervisor_calls) == 1


def test_page_navigate_to_private_url_blocked_before_cdp(monkeypatch):
    calls = []

    monkeypatch.setattr(
        browser_cdp_tool,
        "_resolve_cdp_endpoint",
        lambda: "ws://127.0.0.1:9222/devtools/browser/mock",
    )

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return {"frameId": "f"}

    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", fake_call)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.navigate",
            params={"url": PRIVATE_URL},
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert calls == []


def test_private_guard_inactive_does_not_probe(monkeypatch, cdp_server):
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "ok"}})

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(task_id):
        raise AssertionError("_current_page_private_url must not be probed")

    monkeypatch.setattr(bt, "_current_page_private_url", fail_probe)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            task_id="task-1",
        )
    )

    assert result["success"] is True
    assert result["result"]["result"]["value"] == "ok"


# ---------------------------------------------------------------------------
# check_fn gating
# ---------------------------------------------------------------------------


def test_check_fn_does_not_probe_network(monkeypatch):
    """The availability gate must never hit the network: a stale/unreachable
    configured endpoint used to cost multiple blocking HTTP probes at every
    CLI/Desktop startup (tool-schema assembly), stalling launch by 10+ s."""
    import tools.browser_tool as bt

    def _boom(*a, **k):  # pragma: no cover — the assertion is that it's unused
        raise AssertionError("check_fn must not perform network I/O")

    monkeypatch.setattr(bt, "check_browser_requirements", lambda: True)
    monkeypatch.setattr(bt.requests, "get", _boom)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    assert browser_cdp_tool._browser_cdp_check() is True


def test_check_fn_false_when_browser_requirements_fail(monkeypatch):
    """Even with a CDP URL, gate closes if the overall browser toolset is
    unavailable (e.g. agent-browser not installed)."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(
        bt, "_get_cdp_override_raw", lambda: "ws://localhost:9222/devtools/browser/x"
    )
    assert browser_cdp_tool._browser_cdp_check() is False
