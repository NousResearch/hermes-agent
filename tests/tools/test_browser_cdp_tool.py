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
import requests
from tools import browser_tool_eval_policy as bt_eval_policy
from tools import browser_tool_install as bt_install
from tools import browser_tool_cdp as bt_cdp


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


def test_screenshot_base64_passes_through_unredacted(cdp_server):
    """The Fernet pattern matches arbitrary spans inside base64 payloads —
    a screenshot whose base64 contains "gAAAA..." must stay byte-identical
    instead of being collapsed to "first6...last4" (#94138)."""
    # Real-world shape: the Fernet pattern fires when "gAAAA" follows a "+"
    # or "/" inside the base64 stream (word-boundary requirement).
    shot_b64 = "iVBORw0KGgoAAAANSUhEUg+" + "gAAAA" + "B" * 60 + "=="
    cdp_server.on(
        "Page.captureScreenshot",
        lambda params, sid: {"data": shot_b64},
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Page.captureScreenshot"))

    assert result["success"] is True
    assert result["result"]["data"] == shot_b64


def test_print_to_pdf_base64_passes_through_unredacted(cdp_server):
    pdf_b64 = "JVBERi0xLjcK/" + "gAAAA" + "C" * 60 + "="
    cdp_server.on(
        "Page.printToPDF",
        lambda params, sid: {"data": pdf_b64},
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Page.printToPDF"))

    assert result["success"] is True
    assert result["result"]["data"] == pdf_b64


def test_binary_payload_flag_keeps_secret_redaction_off_method_list(cdp_server):
    """Fail-closed pin: methods without a binary payload field keep full
    secret redaction; the listed methods pass their payload through."""
    fake_key = "sk-" + "CDPSECRETSTILLREDACTED1234567890"
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": fake_key}},
    )
    cdp_server.on(
        "Page.captureScreenshot",
        lambda params, sid: {"data": "gAAAA" + "B" * 60},
    )

    text_result = json.loads(browser_cdp_tool.browser_cdp(method="Runtime.evaluate"))
    assert "CDPSECRETSTILLREDACTED" not in json.dumps(text_result)

    shot_result = json.loads(
        browser_cdp_tool.browser_cdp(method="Page.captureScreenshot")
    )
    assert shot_result["result"]["data"] == "gAAAA" + "B" * 60


def test_binary_payload_field_sibling_string_still_redacted(cdp_server):
    """Path-scoped exemption: on a binary-bearing result only the payload
    field skips redaction; a sibling string keeps full secret redaction,
    proving the exemption cannot widen to the whole result object."""
    fake_key = "sk-" + "CDPSECRETSIBLING1234567890"
    shot_b64 = "iVBORw0KGgoAAAANSUhEUg+" + "gAAAA" + "B" * 60 + "=="
    cdp_server.on(
        "Page.captureScreenshot",
        lambda params, sid: {"data": shot_b64, "note": fake_key},
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Page.captureScreenshot"))

    assert result["success"] is True
    assert result["result"]["data"] == shot_b64
    assert "CDPSECRETSIBLING" not in json.dumps(result)
    assert result["result"]["note"].startswith("sk-")


def test_get_response_body_base64_discriminator_passes_through(cdp_server):
    """Network.getResponseBody with base64Encoded: true — the body is opaque
    base64 bytes and must remain byte-identical (#94138 review on #94142)."""
    body_b64 = "q9Z7" + "gAAAA" + "B" * 60 + "=="
    cdp_server.on(
        "Network.getResponseBody",
        lambda params, sid: {"body": body_b64, "base64Encoded": True},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Network.getResponseBody")
    )

    assert result["success"] is True
    assert result["result"]["body"] == body_b64


def test_get_response_body_text_discriminator_still_redacts(cdp_server):
    """Same method with base64Encoded: false — the body is text and a real
    secret in it must still be redacted."""
    fake_key = "sk-" + "CDPSECRETBODY1234567890"
    cdp_server.on(
        "Network.getResponseBody",
        lambda params, sid: {"body": f"leak {fake_key} here", "base64Encoded": False},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Network.getResponseBody")
    )

    assert result["success"] is True
    assert "CDPSECRETBODY" not in json.dumps(result)


def test_io_read_base64_discriminator_passes_through(cdp_server):
    """IO.read honors the same discriminator contract for its data field."""
    chunk_b64 = "AAA" + "gAAAA" + "C" * 60 + "="
    cdp_server.on(
        "IO.read",
        lambda params, sid: {"data": chunk_b64, "base64Encoded": True, "eof": True},
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="IO.read"))

    assert result["success"] is True
    assert result["result"]["data"] == chunk_b64
    assert result["result"]["eof"] is True


def test_fetch_get_response_body_base64_discriminator_passes_through(cdp_server):
    """Fetch.getResponseBody pins the same body/base64Encoded contract."""
    body_b64 = "zz7+" + "gAAAA" + "D" * 60 + "=="
    cdp_server.on(
        "Fetch.getResponseBody",
        lambda params, sid: {"body": body_b64, "base64Encoded": True},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Fetch.getResponseBody")
    )

    assert result["success"] is True
    assert result["result"]["body"] == body_b64


def test_runtime_evaluate_spoofed_base64_flag_still_redacts(cdp_server):
    """base64Encoded is trusted ONLY on the protocol-defined carrier paths.
    A Runtime.evaluate by-value object carrying
    {"base64Encoded": true, "data": "<secret>"} is untrusted nested JSON —
    the secret must still be redacted (second review on #94142)."""
    fake_key = "sk-" + "CDPSPOOFEDFLAG1234567890"
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {
            "result": {
                "type": "object",
                "value": {"base64Encoded": True, "data": fake_key},
            }
        },
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Runtime.evaluate"))

    assert result["success"] is True
    assert "CDPSPOOFEDFLAG" not in json.dumps(result)


def test_stream_resource_content_unflagged_buffered_data_passes_through(cdp_server):
    """Network.streamResourceContent returns bare binary bufferedData with no
    base64Encoded sibling — declared-binary path, must stay byte-identical."""
    chunk_b64 = "Q2FjaGU/" + "gAAAA" + "E" * 60 + "=="
    cdp_server.on(
        "Network.streamResourceContent",
        lambda params, sid: {"bufferedData": chunk_b64},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Network.streamResourceContent")
    )

    assert result["success"] is True
    assert result["result"]["bufferedData"] == chunk_b64


def test_get_request_post_data_flagged_passes_through(cdp_server):
    """Network.getRequestPostData's postData honors its base64Encoded
    discriminator on the trusted result path."""
    post_b64 = "cG9zdA==" + "gAAAA" + "F" * 60 + "="
    cdp_server.on(
        "Network.getRequestPostData",
        lambda params, sid: {"postData": post_b64, "base64Encoded": True},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Network.getRequestPostData")
    )

    assert result["success"] is True
    assert result["result"]["postData"] == post_b64


def test_get_request_post_data_unflagged_still_redacts(cdp_server):
    fake_key = "sk-" + "CDPPOSTDATASECRET1234567890"
    cdp_server.on(
        "Network.getRequestPostData",
        lambda params, sid: {
            "postData": f"leak {fake_key} here",
            "base64Encoded": False,
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Network.getRequestPostData")
    )

    assert result["success"] is True
    assert "CDPPOSTDATASECRET" not in json.dumps(result)


def test_nested_unflagged_binary_path_passes_through(cdp_server):
    """CacheStorage.requestCachedResponse.response.body is a nested binary
    carrier — the path must exempt the nested field while unrelated nested
    text keeps redaction."""
    fake_key = "sk-" + "CDPNESTEDSECRET1234567890"
    body_b64 = "SUNBRQ/" + "gAAAA" + "G" * 60 + "=="
    cdp_server.on(
        "CacheStorage.requestCachedResponse",
        lambda params, sid: {
            "response": {
                "url": "https://example.test/x",
                "body": body_b64,
                "note": fake_key,
            }
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(method="CacheStorage.requestCachedResponse")
    )

    assert result["success"] is True
    assert result["result"]["response"]["body"] == body_b64
    assert "CDPNESTEDSECRET" not in json.dumps(result)


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


    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", lambda task_id: PRIVATE_URL)

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


    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", lambda task_id: PRIVATE_URL)

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


    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", lambda task_id: None)

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


    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)

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


    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(task_id):
        raise AssertionError("_current_page_private_url must not be probed")

    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", fail_probe)

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

    def _boom(*a, **k):  # pragma: no cover — the assertion is that it's unused
        raise AssertionError("check_fn must not perform network I/O")

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    assert browser_cdp_tool._browser_cdp_check() is True


def test_check_fn_false_when_browser_requirements_fail(monkeypatch):
    """Even with a CDP URL, gate closes if the overall browser toolset is
    unavailable (e.g. agent-browser not installed)."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(
        bt_cdp, "_get_cdp_override_raw", lambda: "ws://localhost:9222/devtools/browser/x"
    )
    assert browser_cdp_tool._browser_cdp_check() is False


# ---------------------------------------------------------------------------
# _resolve_cdp_endpoint
# ---------------------------------------------------------------------------


def test_resolve_cdp_endpoint_returns_stripped_url(monkeypatch):
    """_resolve_cdp_endpoint strips whitespace from the override."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_get_cdp_override", lambda: "  ws://localhost:9222  ")
    assert browser_cdp_tool._resolve_cdp_endpoint() == "ws://localhost:9222"


def test_resolve_cdp_endpoint_returns_empty_when_none(monkeypatch):
    """_resolve_cdp_endpoint returns '' when override is None."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_get_cdp_override", lambda: None)
    assert browser_cdp_tool._resolve_cdp_endpoint() == ""


# ---------------------------------------------------------------------------
# _run_async with running loop
# ---------------------------------------------------------------------------


def test_run_async_with_running_loop():
    """_run_async uses a thread pool when called from inside a running loop."""
    import asyncio

    async def _outer():
        # We're inside a running loop — _run_async should detect this
        # and use a ThreadPoolExecutor.
        async def _inner():
            return 42

        return browser_cdp_tool._run_async(_inner())

    result = asyncio.run(_outer())
    assert result == 42


# ---------------------------------------------------------------------------
# Exception handlers in browser_cdp
# ---------------------------------------------------------------------------


def test_timeout_error_handler(cdp_server):
    """TimeoutError from _cdp_call is caught and returned as tool_error."""
    # Register a handler that sleeps longer than the timeout
    def slow(params, sid):
        time.sleep(10)
        return {}

    cdp_server.on("Page.slowMethod", slow)
    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Page.slowMethod", timeout=0.5)
    )
    assert "error" in result
    # Either "timed out" or "Timed out" — both are TimeoutError paths
    assert "tim" in result["error"].lower()


def test_websocket_error_handler(monkeypatch):
    """WebSocketException is caught and returns a helpful error."""
    monkeypatch.setattr(
        browser_cdp_tool, "_resolve_cdp_endpoint",
        lambda: "ws://localhost:1/devtools/browser/fake",
    )
    result = json.loads(
        browser_cdp_tool.browser_cdp(method="Target.getTargets", timeout=2.0)
    )
    assert "error" in result
    # Connection refused → WebSocket error or generic error
    assert result.get("method") == "Target.getTargets"


# ---------------------------------------------------------------------------
# frame_id routing — supervisor error paths
# ---------------------------------------------------------------------------


def test_frame_id_no_supervisor_attached():
    """browser_cdp with frame_id but no supervisor → helpful error."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task.clear()

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            frame_id="frame-1",
            task_id="default",
        )
    )
    assert "error" in result
    assert "No CDP supervisor" in result["error"]


def test_frame_id_not_found_in_supervisor():
    """browser_cdp with frame_id not in supervisor state → error."""
    from types import SimpleNamespace
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    snap = SimpleNamespace(frame_tree={"top": {"frame_id": "other"}, "children": []})
    supervisor = SimpleNamespace(
        snapshot=lambda: snap,
        _state_lock=__import__("threading").Lock(),
        _frames={},
        _loop=None,
    )
    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

    try:
        result = json.loads(
            browser_cdp_tool.browser_cdp(
                method="Runtime.evaluate",
                frame_id="nonexistent",
                task_id="default",
            )
        )
        assert "error" in result
        assert "not found" in result["error"]
    finally:
        with SUPERVISOR_REGISTRY._lock:
            SUPERVISOR_REGISTRY._by_task.clear()


def test_frame_id_same_origin_no_session():
    """browser_cdp with frame_id that has no session_id → error about OOPIF."""
    from types import SimpleNamespace
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    snap = SimpleNamespace(
        frame_tree={"top": {"frame_id": "top-1"}, "children": [{"frame_id": "child-1"}]}
    )
    supervisor = SimpleNamespace(
        snapshot=lambda: snap,
        _state_lock=__import__("threading").Lock(),
        _frames={},
        _loop=None,
    )
    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

    try:
        result = json.loads(
            browser_cdp_tool.browser_cdp(
                method="Runtime.evaluate",
                frame_id="child-1",
                task_id="default",
            )
        )
        assert "error" in result
        assert "out-of-process" in result["error"]
    finally:
        with SUPERVISOR_REGISTRY._lock:
            SUPERVISOR_REGISTRY._by_task.clear()


def test_frame_id_supervisor_loop_not_running():
    """browser_cdp with frame_id but supervisor loop is None → error."""
    from types import SimpleNamespace
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    snap = SimpleNamespace(
        frame_tree={"top": {}, "children": [{"frame_id": "f1", "session_id": "s1"}]}
    )
    supervisor = SimpleNamespace(
        snapshot=lambda: snap,
        _state_lock=__import__("threading").Lock(),
        _frames={},
        _loop=None,
    )
    with SUPERVISOR_REGISTRY._lock:
        SUPERVISOR_REGISTRY._by_task["default"] = supervisor

    try:
        result = json.loads(
            browser_cdp_tool.browser_cdp(
                method="Runtime.evaluate",
                frame_id="f1",
                task_id="default",
            )
        )
        assert "error" in result
        assert "not running" in result["error"]
    finally:
        with SUPERVISOR_REGISTRY._lock:
            SUPERVISOR_REGISTRY._by_task.clear()
