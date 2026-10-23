"""Unit tests for browser_cdp tool.

Uses a tiny in-process ``websockets`` server to simulate a CDP endpoint —
gives real protocol coverage (connect, send, recv, close) without needing
a real Chrome instance.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List

import pytest

import websockets
from websockets.asyncio.server import serve

from tools import browser_cdp_tool
import requests
from tools import browser_session_cdp
from tools import browser_url_policy
from tools import browser_use_cli


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
        self._handlers: dict[str, Any] = {}
        self._responses: list[dict[str, Any]] = []
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

    def received(self) -> list[dict[str, Any]]:
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


    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)
    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", lambda endpoint, target_id: PRIVATE_URL)

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


    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)
    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", lambda endpoint, target_id: PRIVATE_URL)

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


    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)
    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", lambda endpoint, target_id: None)

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


    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)

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


@pytest.mark.parametrize(("page_url", "blocked"), [(PRIVATE_URL, True), ("https://93.184.216.34/", False)])
def test_page_url_probe_reads_the_target_over_the_same_endpoint(monkeypatch, cdp_server, page_url, blocked):
    """The private-page probe reads the acted-on page's URL from Target.getTargets on the same CDP
    endpoint: a private page refuses page-reading methods before they are sent; a public one runs."""
    cdp_server.on("Target.getTargets", lambda params, sid: {"targetInfos": [
        {"targetId": "other", "type": "page", "url": "https://example.org/"},
        {"targetId": "tab-1", "type": "page", "url": page_url},
    ]})
    cdp_server.on("Target.attachToTarget", lambda params, sid: {"sessionId": "s-1"})
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "page text"}})
    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)

    result = json.loads(browser_cdp_tool.browser_cdp(
        method="Runtime.evaluate", params={"expression": "document.body.innerText"}, target_id="tab-1"))

    evaluated = [m for m in cdp_server.received() if m.get("method") == "Runtime.evaluate"]
    if blocked:
        assert PRIVATE_URL in result["error"] and "private or internal address" in result["error"]
        assert evaluated == []
    else:
        assert result["success"] is True
        assert len(evaluated) == 1


PUBLIC_PAGE = "https://93.184.216.34/"


@pytest.mark.parametrize(("target_id", "page_url", "blocked"), [
    ("worker-1", PRIVATE_URL, True),      # SharedWorker of a private page: fetch('/') reads that page
    ("bogus-id", PRIVATE_URL, True),      # unknown target: cannot prove it is not the private page
    (None, PRIVATE_URL, True),            # browser-level call spans every page
    ("worker-1", PUBLIC_PAGE, False),     # every page public: the worker runs
])
def test_non_page_target_fails_closed_while_any_page_is_private(monkeypatch, cdp_server, target_id, page_url,
                                                                 blocked):
    """A non-page / unknown target carries some page's origin, so page-reading CDP refuses it while any page
    target is private, and runs it when every page is public."""
    cdp_server.on("Target.getTargets", lambda params, sid: {"targetInfos": [
        {"targetId": "tab-public", "type": "page", "url": PUBLIC_PAGE},
        {"targetId": "tab-2", "type": "page", "url": page_url},
        {"targetId": "newtab", "type": "page", "url": "chrome://newtab/"},
        {"targetId": "worker-1", "type": "shared_worker", "url": ""},
    ]})
    cdp_server.on("Target.attachToTarget", lambda params, sid: {"sessionId": "s-1"})
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "page body"}})
    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)

    result = json.loads(browser_cdp_tool.browser_cdp(
        method="Runtime.evaluate", params={"expression": "fetch('/').then(r => r.text())"}, target_id=target_id))

    evaluated = [m for m in cdp_server.received() if m.get("method") == "Runtime.evaluate"]
    if blocked:
        assert PRIVATE_URL in result["error"] and evaluated == []
    else:
        assert result["success"] is True and len(evaluated) == 1


def test_frame_id_is_judged_by_its_own_target_not_the_first_page(monkeypatch, cdp_server):
    """frame_id routing checks the OOPIF's own target (frame id == iframe target id) and fails closed on a
    private page, even when the first page target is public."""
    supervisor_calls = []
    cdp_server.on("Target.getTargets", lambda params, sid: {"targetInfos": [
        {"targetId": "tab-public", "type": "page", "url": PUBLIC_PAGE},
        {"targetId": "tab-private", "type": "page", "url": PRIVATE_URL},
        {"targetId": "frame-1", "type": "iframe", "url": "https://93.184.216.35/embed"},
    ]})
    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)
    monkeypatch.setattr(browser_cdp_tool, "_browser_cdp_via_supervisor",
                        lambda **kw: supervisor_calls.append(kw) or json.dumps({"success": True}))

    result = json.loads(browser_cdp_tool.browser_cdp(
        method="Runtime.evaluate", params={"expression": "document.title"}, frame_id="frame-1"))

    assert PRIVATE_URL in result["error"] and supervisor_calls == []


def test_unreadable_target_list_fails_closed(monkeypatch, cdp_server):
    """When the page URLs cannot be read the guard refuses instead of assuming the page is public."""
    cdp_server.on("Target.getTargets", lambda params, sid: RuntimeError("Target domain unavailable"))
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "page body"}})
    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)

    result = json.loads(browser_cdp_tool.browser_cdp(
        method="Runtime.evaluate", params={"expression": "document.title"}))

    assert "could not read the browser's targets" in result["error"]
    assert not [m for m in cdp_server.received() if m.get("method") == "Runtime.evaluate"]


LOCAL_TAB = "http://127.0.0.1:3000/"


@pytest.mark.parametrize(("method", "params", "allowed"), [
    ("Target.createTarget", {"url": PUBLIC_PAGE}, True),
    ("Target.createTarget", {"url": "http://127.0.0.1:3000/admin"}, False),  # checked like Page.navigate
    ("Target.createTarget", {"url": PRIVATE_URL}, False),
    ("Target.createTarget", {"url": "about:blank"}, True),  # a blank tab, as Hermes's own helpers open
    ("Target.createTarget", {"url": ""}, True),
    ("Target.createTarget", {}, True),
    ("Target.createTarget", {"url": "file:///etc/passwd"}, False),  # every other non-http scheme stays refused
    ("Target.createTarget", {"url": "about:blank@127.0.0.1"}, False),
    ("Target.createTarget", {"url": "data:text/html,<script>fetch('http://127.0.0.1:3000/')</script>"}, False),
    ("Target.activateTarget", {"targetId": "tab-public"}, True),
    ("Target.closeTarget", {"targetId": "tab-public"}, True),
    ("Network.getAllCookies", {}, False),  # every other target-less call stays fail-closed
    ("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": "/srv"}, False),
])
def test_tab_management_stays_usable_while_a_tab_is_private(monkeypatch, cdp_server, method, params, allowed):
    """One private tab (a local dev server) must not stop the model opening, switching or closing tabs; a
    new tab is blank or its url is held to Page.navigate's rule, and every other browser-level call is refused."""
    cdp_server.on("Target.getTargets", lambda params, sid: {"targetInfos": [
        {"targetId": "tab-public", "type": "page", "url": PUBLIC_PAGE},
        {"targetId": "tab-local", "type": "page", "url": LOCAL_TAB},
    ]})
    cdp_server.on(method, lambda params, sid: {"targetId": "new-tab"} if method == "Target.createTarget" else {})
    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: True)

    result = json.loads(browser_cdp_tool.browser_cdp(method=method, params=params))

    sent = [m for m in cdp_server.received() if m.get("method") == method]
    if allowed:
        assert result["success"] is True and len(sent) == 1
    else:
        assert "private or internal address" in result["error"] and sent == []


def test_private_guard_inactive_does_not_probe(monkeypatch, cdp_server):
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "ok"}})


    monkeypatch.setattr(browser_url_policy, "eval_ssrf_guard_active", lambda: False)

    def fail_probe(endpoint, target_id):
        raise AssertionError("_current_page_private_url must not be probed")

    monkeypatch.setattr(browser_cdp_tool, "_current_page_private_url", fail_probe)

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

    monkeypatch.setattr(browser_use_cli, "browser_exec_available", lambda: True)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    assert browser_cdp_tool._browser_cdp_check() is True


def test_check_fn_false_when_browser_requirements_fail(monkeypatch):
    """Even with a CDP URL, gate closes if the overall browser surface is
    unavailable (browser-harness not importable)."""

    monkeypatch.setattr(browser_use_cli, "browser_exec_available", lambda: False)
    monkeypatch.setattr(
        browser_session_cdp, "get_cdp_override_raw", lambda: "ws://localhost:9222/devtools/browser/x"
    )
    assert browser_cdp_tool._browser_cdp_check() is False
