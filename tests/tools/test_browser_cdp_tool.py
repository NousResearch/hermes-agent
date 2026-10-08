"""Unit tests for browser_cdp tool.

Uses a tiny in-process ``websockets`` server to simulate a CDP endpoint —
gives real protocol coverage (connect, send, recv, close) without needing
a real Chrome instance.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, cast

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


@pytest.mark.parametrize(
    ("selected_url", "guard_active", "blocked"),
    [
        (PRIVATE_URL, True, True),
        ("https://example.test/page", True, False),
        (PRIVATE_URL, False, False),
    ],
)
def test_target_id_guard_uses_selected_target_url(
    monkeypatch, cdp_server, selected_url, guard_active, blocked
):
    cdp_server.on(
        "Target.getTargetInfo",
        lambda params, sid: {
            "targetInfo": {
                "targetId": params["targetId"],
                "type": "page",
                "url": selected_url,
            }
        },
    )
    cdp_server.on(
        "Target.attachToTarget", lambda params, sid: {"sessionId": "selected-session"}
    )
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": "selected data"}},
    )
    monkeypatch.setattr(
        bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: guard_active
    )
    monkeypatch.setattr(
        bt_eval_policy, "_current_page_private_url", lambda task_id: None
    )
    monkeypatch.setattr(
        bt_eval_policy, "_url_blocked", lambda bt, url: url == PRIVATE_URL
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            target_id="selected-target",
            task_id="task-1",
        )
    )
    requests = cdp_server.received()
    methods = [request["method"] for request in requests]

    assert ("error" in result) is blocked
    if blocked:
        assert PRIVATE_URL in result["error"]
    else:
        assert result["success"] is True
        assert "Runtime.evaluate" in methods
    target_info_requests = [
        request for request in requests if request["method"] == "Target.getTargetInfo"
    ]
    assert len(target_info_requests) == int(guard_active)
    if target_info_requests:
        assert target_info_requests[0]["params"] == {"targetId": "selected-target"}
        assert "sessionId" not in target_info_requests[0]
    if blocked:
        assert "Runtime.evaluate" not in methods


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


@pytest.mark.parametrize(
    ("selected_url", "guard_active", "blocked"),
    [
        (PRIVATE_URL, True, True),
        ("https://example.test/frame", True, False),
        (PRIVATE_URL, False, False),
    ],
)
def test_frame_id_guard_uses_selected_frame_url(
    monkeypatch, selected_url, guard_active, blocked
):
    from agent import async_utils
    from tools.browser_supervisor import CDPSupervisor, SUPERVISOR_REGISTRY
    from tools.browser_supervisor_frames import FrameInfo

    supervisor = CDPSupervisor("task-1", "ws://127.0.0.1/devtools/browser/mock")
    supervisor._set_frame(
        FrameInfo("top-frame", "https://ambient.test/", "", None, False, "top-session")
    )
    supervisor._set_frame(
        FrameInfo(
            "selected-frame",
            selected_url,
            "",
            "top-frame",
            True,
            "selected-session",
        )
    )
    cdp_calls = []

    async def fake_cdp(method, params, *, session_id, timeout):
        cdp_calls.append((method, params, session_id, timeout))
        return {"result": {"result": {"type": "string", "value": "selected data"}}}

    class RunningLoop:
        def is_running(self):
            return True

    class ImmediateFuture:
        def __init__(self, coro):
            self.coro = coro

        def result(self, timeout):
            return asyncio.run(self.coro)

    supervisor._loop = cast(Any, RunningLoop())
    monkeypatch.setattr(supervisor, "_cdp", fake_cdp)
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)
    monkeypatch.setattr(
        async_utils,
        "safe_schedule_threadsafe",
        lambda coro, loop: ImmediateFuture(coro),
    )
    monkeypatch.setattr(
        bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: guard_active
    )
    monkeypatch.setattr(
        bt_eval_policy, "_current_page_private_url", lambda task_id: None
    )
    monkeypatch.setattr(
        bt_eval_policy, "_url_blocked", lambda bt, url: url == PRIVATE_URL
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="selected-frame",
            task_id="task-1",
        )
    )

    assert ("error" in result) is blocked
    if blocked:
        assert PRIVATE_URL in result["error"]
        assert cdp_calls == []
    else:
        assert result["success"] is True
        assert cdp_calls == [
            (
                "Runtime.evaluate",
                {"expression": "document.body.innerText"},
                "selected-session",
                30.0,
            )
        ]


# Selected-destination matrix (review on #128503): the REAL page-URL predicate runs here —
# only DNS is faked, so the scheme floor and the SSRF verdict are both exercised.
SELECTED_DESTINATION_MATRIX = [
    # (selected page/frame URL, blocked?)
    ("https://example.com/", False),
    ("http://127.0.0.1:9222/", True),
    ("http://LocalHost.:9222/json", True),
    ("http://[::ffff:127.0.0.1]:9222/", True),
    ("http://metadata.google.internal/", True),
    ("about:blank", False),
    ("chrome://newtab/", False),
    ("devtools://devtools/bundled/inspector.html", False),
    ("chrome-extension://abcdefghijklmnop/popup.html", False),
    ("data:text/html,<h1>hi", False),
    ("blob:https://example.com/2f1a", False),
    ("blob:null/2f1a", False),
    ("view-source:https://example.com", False),
    # Wrappers are judged by the URL they wrap, so they cannot launder a private page.
    ("blob:http://127.0.0.1:9222/2f1a", True),
    ("view-source:http://127.0.0.1:9222/json", True),
    ("VIEW-SOURCE:http://10.0.0.5/", True),
    ("view-source:blob:http://192.168.1.1/2f1a", True),
    ("filesystem:http://10.0.0.5/temporary/x", True),
    # file: is a genuine local read on the browser host: blocked explicitly, also when wrapped.
    ("file:///tmp/report.html", True),
    ("view-source:file:///etc/passwd", True),
    # Other authority-bearing schemes are host-checked, not waved through.
    ("ftp://10.0.0.1/pub/", True),
    ("ftp://example.com/pub/", False),
]


@pytest.fixture
def fake_public_dns(monkeypatch):
    import ipaddress
    import socket
    from tools import url_safety

    def fake_getaddrinfo(hostname, port=None):
        try:
            ip = str(ipaddress.ip_address(hostname))
        except ValueError:
            ip = "127.0.0.1" if hostname.rstrip(".").lower() == "localhost" else "93.184.215.14"
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(url_safety, "_getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(url_safety, "_global_allow_private_urls", lambda: False)


@pytest.mark.parametrize(("selected_url", "blocked"), SELECTED_DESTINATION_MATRIX)
def test_page_url_predicate_matrix(fake_public_dns, selected_url, blocked):
    from tools import browser_tool as bt

    assert bt_eval_policy._page_url_blocked(bt, selected_url) is blocked
    # The navigate-time predicate stays fail-closed for every non-http(s) scheme.
    if not selected_url.lower().startswith(("http:", "https:")):
        assert bt_eval_policy._url_blocked(bt, selected_url) is True


@pytest.mark.parametrize(("selected_url", "blocked"), SELECTED_DESTINATION_MATRIX)
def test_target_id_guard_matrix_with_real_predicate(
    monkeypatch, cdp_server, fake_public_dns, selected_url, blocked
):
    cdp_server.on(
        "Target.getTargetInfo",
        lambda params, sid: {
            "targetInfo": {"targetId": params["targetId"], "type": "page", "url": selected_url}
        },
    )
    cdp_server.on("Target.attachToTarget", lambda params, sid: {"sessionId": "selected-session"})
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": "selected data"}},
    )
    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", lambda task_id: None)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            target_id="selected-target",
            task_id="task-1",
        )
    )
    methods = [request["method"] for request in cdp_server.received()]

    if blocked:
        assert selected_url in result["error"]
        assert "Runtime.evaluate" not in methods
    else:
        assert result.get("success") is True, result
        assert "Runtime.evaluate" in methods


def test_supervisor_created_about_blank_target_is_usable(monkeypatch, cdp_server, fake_public_dns):
    """The supervisor opens its own page via Target.createTarget(url='about:blank'); the
    next raw CDP call against that target must not be refused as 'private'."""
    cdp_server.on(
        "Target.getTargetInfo",
        lambda params, sid: {"targetInfo": {"targetId": params["targetId"], "type": "page", "url": "about:blank"}},
    )
    cdp_server.on("Target.attachToTarget", lambda params, sid: {"sessionId": "s"})
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"type": "number", "value": 2}})
    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt_eval_policy, "_current_page_private_url", lambda task_id: None)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate", params={"expression": "1+1"}, target_id="fresh", task_id="task-1"
        )
    )

    assert result.get("success") is True, result


def test_ambient_page_guard_allows_about_blank_but_not_wrapped_private(monkeypatch, fake_public_dns):
    """The ambient current-page probe shares the already-loaded-page predicate."""
    from tools import browser_tool_session

    current = {"url": "about:blank"}
    monkeypatch.setattr(
        browser_tool_session,
        "_run_browser_command",
        lambda *a, **k: {"success": True, "data": {"result": current["url"]}},
    )

    assert bt_eval_policy._current_page_private_url("task-1") is None
    current["url"] = "view-source:http://127.0.0.1:9222/json"
    assert bt_eval_policy._current_page_private_url("task-1") == current["url"]


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

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)
    monkeypatch.setattr(
        bt_cdp, "_get_cdp_override_raw", lambda: "ws://localhost:9222/devtools/browser/x"
    )
    assert browser_cdp_tool._browser_cdp_check() is False
