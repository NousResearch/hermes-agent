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
                                events_before = []
                                events_after = []
                                if isinstance(result, dict):
                                    result = dict(result)
                                    events_before = result.pop("__events_before__", [])
                                    events_after = result.pop("__events_after__", [])
                                for event in events_before:
                                    await ws.send(json.dumps(event))
                                reply = {"id": call_id, "result": result}
                            except Exception as exc:
                                reply = {
                                    "id": call_id,
                                    "error": {"code": -1, "message": str(exc)},
                                }
                        if session_id:
                            reply["sessionId"] = session_id
                        await ws.send(json.dumps(reply))
                        for event in events_after:
                            await ws.send(json.dumps(event))
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


def test_browser_level_redacts_secret_result(cdp_server, monkeypatch):
    import agent.redact

    secret_key_one = "sk-" + "CDPSECRETKEYONE1234567890"
    secret_key_two = "sk-" + "CDPSECRETKEYTWO1234567890"
    fake_key = "sk-" + "CDPSECRETRESULT1234567890"
    real_redact = agent.redact.redact_sensitive_text

    def colliding_redact(text, **kwargs):
        if text in {secret_key_one, secret_key_two}:
            return "sk-«redacted:test…»"
        return real_redact(text, **kwargs)

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", colliding_redact)
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {
            "result": {
                "type": "object",
                "value": fake_key,
                "properties": {
                    secret_key_one: {"source": "first"},
                    secret_key_two: {"source": "second"},
                },
            }
        },
    )

    result = json.loads(browser_cdp_tool.browser_cdp(method="Runtime.evaluate"))

    assert result["success"] is True
    serialized = json.dumps(result)
    assert "CDPSECRETRESULT" not in serialized
    assert "CDPSECRETKEY" not in serialized
    assert result["result"]["result"]["value"].startswith("sk-")
    properties = result["result"]["result"]["properties"]
    assert len(properties) == 2
    assert {entry["source"] for entry in properties.values()} == {"first", "second"}


def test_cdp_key_redaction_preserves_large_collision_sets(monkeypatch):
    import agent.redact

    key_count = 2_000
    payload = {f"secret-{index}": index for index in range(key_count)}
    monkeypatch.setattr(
        agent.redact,
        "redact_sensitive_text",
        lambda text, **kwargs: "redacted" if text.startswith("secret-") else text,
    )

    first = browser_cdp_tool._redact_cdp_output(payload)
    second = browser_cdp_tool._redact_cdp_output(payload)

    assert first == second
    assert len(first) == key_count
    assert set(first.values()) == set(range(key_count))
    assert list(first)[:3] == ["redacted", "redacted (2)", "redacted (3)"]
    assert list(first)[-1] == f"redacted ({key_count})"


# ---------------------------------------------------------------------------
# Happy-path: target-attached call
# ---------------------------------------------------------------------------


def test_public_top_level_target_is_resolved_before_attach(cdp_server, monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: True)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "public-page", "type": "page", "url": "https://example.com"}
            ]
        },
    )
    cdp_server.on(
        "Target.attachToTarget", lambda params, sid: {"sessionId": "page-session"}
    )
    cdp_server.on(
        "Page.getFrameTree",
        lambda params, sid: {
            "frameTree": {"frame": {"url": "https://example.com"}}
        },
    )
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": "public"}},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            target_id="public-page",
            task_id="task-1",
        )
    )

    assert result["success"] is True
    assert [call["method"] for call in cdp_server.received()] == [
        "Target.getTargets",
        "Target.attachToTarget",
        "Page.getFrameTree",
        "Runtime.evaluate",
    ]


def test_private_oopif_target_id_is_blocked_before_attach(cdp_server, monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "private-oopif", "type": "iframe", "url": PRIVATE_URL}
            ]
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            target_id="private-oopif",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "top-level page target" in result["error"]
    assert [call["method"] for call in cdp_server.received()] == ["Target.getTargets"]


def test_non_page_target_is_rejected_when_ssrf_guard_is_inactive(cdp_server, monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: False)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "child-target", "type": "iframe", "url": "https://child.example"}
            ]
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            target_id="child-target",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "top-level page target" in result["error"]
    assert [call["method"] for call in cdp_server.received()] == ["Target.getTargets"]


def test_allowlisted_method_still_rejects_non_page_target(cdp_server, monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "private-oopif", "type": "iframe", "url": PRIVATE_URL}
            ]
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.reload",
            target_id="private-oopif",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "top-level page target" in result["error"]
    assert [call["method"] for call in cdp_server.received()] == ["Target.getTargets"]


def test_target_navigation_to_private_url_is_blocked_after_attach(cdp_server, monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: url != PRIVATE_URL)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "changing-page", "type": "page", "url": "https://example.com"}
            ]
        },
    )
    cdp_server.on(
        "Target.attachToTarget", lambda params, sid: {"sessionId": "page-session"}
    )
    cdp_server.on(
        "Page.getFrameTree",
        lambda params, sid: {"frameTree": {"frame": {"url": PRIVATE_URL}}},
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            target_id="changing-page",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "navigated to a private or internal address" in result["error"]
    assert [call["method"] for call in cdp_server.received()] == [
        "Target.getTargets",
        "Target.attachToTarget",
        "Page.getFrameTree",
    ]


@pytest.mark.parametrize(
    ("method", "params", "selected_frame_id", "private_commit_only"),
    [
        (
            "Page.navigate",
            {"url": "https://public.example/redirect"},
            "top-frame",
            False,
        ),
        ("Page.reload", {}, "top-frame", False),
        (
            "Page.navigate",
            {"url": "https://public.example/redirect", "frameId": "child-frame"},
            "child-frame",
            False,
        ),
        (
            "Page.navigate",
            {"url": "https://public.example/private-then-public"},
            "top-frame",
            True,
        ),
    ],
)
def test_target_navigation_result_is_revalidated_and_blank_on_private_redirect(
    cdp_server, monkeypatch, method, params, selected_frame_id, private_commit_only
):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: url != PRIVATE_URL)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "public-page", "type": "page", "url": "https://public.example"}
            ]
        },
    )
    cdp_server.on(
        "Target.attachToTarget", lambda params, sid: {"sessionId": "page-session"}
    )
    cdp_server.on("Page.enable", lambda params, sid: {})
    frame_tree_calls = 0

    def frame_tree(params, sid):
        nonlocal frame_tree_calls
        frame_tree_calls += 1
        selected_url = (
            "https://public.example"
            if frame_tree_calls == 1 or private_commit_only
            else PRIVATE_URL
        )
        child_frames = []
        top_url = selected_url
        if selected_frame_id == "child-frame":
            top_url = "https://public.example"
            child_frames = [
                {
                    "frame": {
                        "id": "child-frame",
                        "loaderId": "initial-child-loader",
                        "url": selected_url,
                    }
                }
            ]
        return {
            "frameTree": {
                "frame": {
                    "id": "top-frame",
                    "loaderId": "initial-loader",
                    "url": top_url,
                },
                "childFrames": child_frames,
            }
        }

    def navigate(params, sid):
        if params.get("url") == "about:blank":
            return {
                "frameId": "top-frame",
                "loaderId": "blank-loader",
                "__events_after__": [
                    {
                        "method": "Page.frameNavigated",
                        "params": {
                            "frame": {
                                "id": "top-frame",
                                "loaderId": "blank-loader",
                                "url": "about:blank",
                            }
                        },
                    }
                ],
            }
        return navigation_result()

    def navigation_result():
        return {
            "frameId": selected_frame_id,
            "loaderId": "private-loader",
            "__events_after__": [
                {
                    "method": "Page.frameNavigated",
                    "params": {
                        "frame": {
                            "id": selected_frame_id,
                            "loaderId": "private-loader",
                            "url": PRIVATE_URL,
                        }
                    },
                }
            ],
        }

    cdp_server.on(method, lambda params, sid: navigation_result())
    cdp_server.on(
        "Page.getFrameTree",
        frame_tree,
    )
    cdp_server.on("Page.navigate", navigate)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method=method,
            params=params,
            target_id="public-page",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "landed on a private or internal address" in result["error"]
    calls = cdp_server.received()
    expected = [
        "Target.getTargets",
        "Target.attachToTarget",
    ]
    if method == "Page.reload":
        # Reload is not allowlisted, so the selected-target private-page
        # revalidation runs immediately after attach.
        expected.append("Page.getFrameTree")
    expected.extend(
        [
            "Page.enable",
            "Page.getFrameTree",
            method,
            "Page.getFrameTree",
            "Page.navigate",
        ]
    )
    assert [call["method"] for call in calls] == expected
    assert calls[-1]["params"] == {"url": "about:blank"}


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


def test_frame_id_rejects_non_dict_params_before_guard(monkeypatch):
    """frame_id path must reject non-dict params like the stateless path.

    Without this check, a truthy non-dict (e.g. a string) reaches
    ``_browser_cdp_private_guard``, raises on ``.get``, and the guard's
    broad except fail-opens — skipping the private-page boundary and
    never surfacing the clear params validation error.
    """
    supervisor_calls = []
    guard_calls = []

    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: PRIVATE_URL)

    real_guard = browser_cdp_tool._browser_cdp_private_guard

    def tracking_guard(**kwargs):
        guard_calls.append(kwargs)
        return real_guard(**kwargs)

    def fake_supervisor_route(**kwargs):
        supervisor_calls.append(kwargs)
        return json.dumps({"success": True, "result": {"value": "leaked"}})

    monkeypatch.setattr(browser_cdp_tool, "_browser_cdp_private_guard", tracking_guard)
    monkeypatch.setattr(
        browser_cdp_tool, "_browser_cdp_via_supervisor", fake_supervisor_route
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params="not-a-dict",
            frame_id="frame-1",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "params" in result["error"].lower()
    assert "object/dict" in result["error"]
    assert "str" in result["error"]
    assert guard_calls == []
    assert supervisor_calls == []


class _FakeFrame:
    def __init__(self, **kwargs):
        self._data = dict(kwargs)

    def to_dict(self):
        return dict(self._data)


class _FakeSnapshot:
    def __init__(self, frame_tree):
        self.frame_tree = frame_tree


class _FakeSupervisor:
    """Minimal supervisor stub for frame_id private-URL routing tests."""

    def __init__(self, *, frame_tree=None, frames=None):
        self._frame_tree = frame_tree or {"top": None, "children": []}
        self._frames = frames or {}
        self._state_lock = threading.Lock()
        self._loop = None
        self.cdp_calls = []

    def snapshot(self):
        return _FakeSnapshot(self._frame_tree)


def test_frame_id_blocks_private_oopif_when_top_page_is_public(monkeypatch):
    """Public parent + private OOPIF child must still block page-content CDP."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    # Top-level page is public — the regression the review called out.
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    supervisor = _FakeSupervisor(
        frame_tree={
            "top": {
                "frame_id": "top-1",
                "url": "https://example.com/",
                "origin": "https://example.com",
                "session_id": "top-session",
            },
            "children": [
                {
                    "frame_id": "oopif-private",
                    "url": PRIVATE_URL,
                    "origin": "http://169.254.169.254",
                    "session_id": "child-session",
                }
            ],
        }
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="oopif-private",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert "private or internal address" in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_raw_frames_fallback_blocks_private_oopif(monkeypatch):
    """Raw ``_frames`` fallback (outside capped frame_tree) must apply the same check."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    supervisor = _FakeSupervisor(
        frame_tree={"top": {"frame_id": "top-1", "url": "https://example.com/"}, "children": []},
        frames={
            "oopif-raw": _FakeFrame(
                frame_id="oopif-raw",
                url=PRIVATE_URL,
                origin="http://169.254.169.254",
                session_id="raw-session",
            )
        },
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="DOM.getDocument",
            params={},
            frame_id="oopif-raw",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_allowlist_survives_private_oopif(monkeypatch):
    """Non-reload allowlist methods must still reach supervisor on private frames."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    private_frame = _FakeFrame(
        frame_id="oopif-private",
        url=PRIVATE_URL,
        origin="http://169.254.169.254",
        session_id="child-session",
    )
    import agent.redact

    secret_key_one = "sk-" + "CDPFRAMEKEYONE1234567890"
    secret_key_two = "sk-" + "CDPFRAMEKEYTWO1234567890"
    fake_key = "sk-" + "CDPFRAMESECRET1234567890"
    real_redact = agent.redact.redact_sensitive_text

    def colliding_redact(text, **kwargs):
        if text in {secret_key_one, secret_key_two}:
            return "sk-«redacted:test…»"
        return real_redact(text, **kwargs)

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", colliding_redact)

    class _AllowlistSupervisor(_FakeSupervisor):
        def __init__(self):
            super().__init__(
                frame_tree={
                    "top": {
                        "frame_id": "top-1",
                        "url": "https://example.com/",
                        "origin": "https://example.com",
                    },
                    "children": [private_frame.to_dict()],
                },
                frames={"oopif-private": private_frame},
            )
            # Running loop stub so via_supervisor proceeds past the loop check.
            self._loop = type(
                "Loop",
                (),
                {"is_running": staticmethod(lambda: True)},
            )()

        async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
            self.cdp_calls.append(
                {"method": method, "params": params, "session_id": session_id}
            )
            return {
                "result": {
                    "value": fake_key,
                    "properties": {
                        secret_key_one: {"source": "first"},
                        secret_key_two: {"source": "second"},
                    },
                }
            }

    supervisor = _AllowlistSupervisor()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    def fake_schedule(coro, loop):
        class _Fut:
            def result(self, timeout=None):
                return asyncio.run(coro)

        return _Fut()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.stopLoading",
            params={},
            frame_id="oopif-private",
            task_id="task-1",
        )
    )

    assert result.get("success") is True
    assert result.get("session_id") == "child-session"
    assert "CDPFRAMESECRET" not in json.dumps(result)
    assert "CDPFRAMEKEY" not in json.dumps(result)
    assert result["result"]["value"].startswith("sk-")
    properties = result["result"]["properties"]
    assert len(properties) == 2
    assert {entry["source"] for entry in properties.values()} == {"first", "second"}
    assert supervisor.cdp_calls == [
        {"method": "Page.stopLoading", "params": {}, "session_id": "child-session"}
    ]


def test_frame_id_reload_blocked_on_private_oopif(monkeypatch):
    """Page.reload must not re-fetch an already-private OOPIF child session."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    private_frame = _FakeFrame(
        frame_id="oopif-private",
        url=PRIVATE_URL,
        origin="http://169.254.169.254",
        session_id="child-session",
    )
    supervisor = _FakeSupervisor(
        frame_tree={
            "top": {
                "frame_id": "top-1",
                "url": "https://example.com/",
                "origin": "https://example.com",
            },
            "children": [private_frame.to_dict()],
        },
        frames={"oopif-private": private_frame},
    )
    supervisor._loop = type(
        "Loop",
        (),
        {"is_running": staticmethod(lambda: True)},
    )()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.reload",
            params={},
            frame_id="oopif-private",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_navigate_redirect_to_private_is_blanked(monkeypatch):
    """frame_id Page.navigate must post-check and blank public-to-private landings."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: url != PRIVATE_URL)

    public_frame = _FakeFrame(
        frame_id="oopif-public",
        url="https://public.example/widget",
        origin="https://public.example",
        session_id="child-session",
        loaderId="public-loader",
    )

    class _NavSupervisor(_FakeSupervisor):
        def __init__(self):
            super().__init__(
                frame_tree={
                    "top": {
                        "frame_id": "top-1",
                        "url": "https://example.com/",
                        "origin": "https://example.com",
                    },
                    "children": [public_frame.to_dict()],
                },
                frames={"oopif-public": public_frame},
            )
            self._loop = type(
                "Loop",
                (),
                {"is_running": staticmethod(lambda: True)},
            )()

        async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
            self.cdp_calls.append(
                {"method": method, "params": params or {}, "session_id": session_id}
            )
            if method == "Page.enable":
                return {"result": {}}
            if method == "Page.navigate" and (params or {}).get("url") != "about:blank":
                public_frame._data["url"] = PRIVATE_URL
                public_frame._data["origin"] = "http://169.254.169.254"
                public_frame._data["loaderId"] = "private-loader"
                return {
                    "result": {
                        "frameId": "oopif-public",
                        "loaderId": "private-loader",
                    }
                }
            if method == "Page.getFrameTree":
                return {
                    "result": {
                        "frameTree": {
                            "frame": {
                                "id": "oopif-public",
                                "url": public_frame._data["url"],
                                "loaderId": public_frame._data.get("loaderId", ""),
                            }
                        }
                    }
                }
            if method == "Page.navigate" and (params or {}).get("url") == "about:blank":
                public_frame._data["url"] = "about:blank"
                public_frame._data["origin"] = "null"
                public_frame._data["loaderId"] = "blank-loader"
                return {"result": {"frameId": "oopif-public", "loaderId": "blank-loader"}}
            return {"result": {}}

    supervisor = _NavSupervisor()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    def fake_schedule(coro, loop):
        class _Fut:
            def result(self, timeout=None):
                return asyncio.run(coro)

        return _Fut()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.navigate",
            params={"url": "https://public.example/redirect"},
            frame_id="oopif-public",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "landed on a private or internal address" in result["error"]
    assert "frame reset to about:blank" in result["error"]
    assert [call["method"] for call in supervisor.cdp_calls] == [
        "Page.enable",
        "Page.getFrameTree",
        "Page.navigate",
        "Page.getFrameTree",
        "Page.navigate",
        "Page.getFrameTree",
    ]
    assert supervisor.cdp_calls[2]["params"] == {"url": "https://public.example/redirect"}
    assert supervisor.cdp_calls[4]["params"] == {"url": "about:blank"}
    assert public_frame._data["url"] == "about:blank"


def test_frame_id_navigate_waits_for_delayed_private_commit(monkeypatch):
    """frame_id navigate must not trust a still-public tree before redirect commit."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: url != PRIVATE_URL)

    public_frame = _FakeFrame(
        frame_id="oopif-public",
        url="https://public.example/widget",
        origin="https://public.example",
        session_id="child-session",
        loaderId="public-loader",
    )
    post_nav_tree_reads = {"count": 0}
    navigated = {"done": False}

    class _DelayedCommitSupervisor(_FakeSupervisor):
        def __init__(self):
            super().__init__(
                frame_tree={
                    "top": {
                        "frame_id": "top-1",
                        "url": "https://example.com/",
                        "origin": "https://example.com",
                    },
                    "children": [public_frame.to_dict()],
                },
                frames={"oopif-public": public_frame},
            )
            self._loop = type(
                "Loop",
                (),
                {"is_running": staticmethod(lambda: True)},
            )()

        async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
            self.cdp_calls.append(
                {"method": method, "params": params or {}, "session_id": session_id}
            )
            if method == "Page.enable":
                return {"result": {}}
            if method == "Page.navigate" and (params or {}).get("url") == "about:blank":
                public_frame._data["url"] = "about:blank"
                public_frame._data["origin"] = "null"
                public_frame._data["loaderId"] = "blank-loader"
                return {"result": {"frameId": "oopif-public", "loaderId": "blank-loader"}}
            if method == "Page.navigate":
                # Navigate returns the new loaderId while the frame is still on
                # the public URL — mirrors CDP returning before redirect commit.
                navigated["done"] = True
                return {
                    "result": {
                        "frameId": "oopif-public",
                        "loaderId": "private-loader",
                    }
                }
            if method == "Page.getFrameTree":
                if public_frame._data.get("loaderId") == "blank-loader":
                    return {
                        "result": {
                            "frameTree": {
                                "frame": {
                                    "id": "oopif-public",
                                    "url": "about:blank",
                                    "loaderId": "blank-loader",
                                }
                            }
                        }
                    }
                if not navigated["done"]:
                    return {
                        "result": {
                            "frameTree": {
                                "frame": {
                                    "id": "oopif-public",
                                    "url": public_frame._data["url"],
                                    "loaderId": public_frame._data.get(
                                        "loaderId", "public-loader"
                                    ),
                                }
                            }
                        }
                    }
                post_nav_tree_reads["count"] += 1
                if post_nav_tree_reads["count"] < 3:
                    # Still the old public commit — early revalidate would pass.
                    return {
                        "result": {
                            "frameTree": {
                                "frame": {
                                    "id": "oopif-public",
                                    "url": "https://public.example/widget",
                                    "loaderId": "public-loader",
                                }
                            }
                        }
                    }
                public_frame._data["url"] = PRIVATE_URL
                public_frame._data["origin"] = "http://169.254.169.254"
                public_frame._data["loaderId"] = "private-loader"
                return {
                    "result": {
                        "frameTree": {
                            "frame": {
                                "id": "oopif-public",
                                "url": PRIVATE_URL,
                                "loaderId": "private-loader",
                            }
                        }
                    }
                }
            return {"result": {}}

    supervisor = _DelayedCommitSupervisor()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    def fake_schedule(coro, loop):
        class _Fut:
            def result(self, timeout=None):
                return asyncio.run(coro)

        return _Fut()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.navigate",
            params={"url": "https://public.example/redirect"},
            frame_id="oopif-public",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "landed on a private or internal address" in result["error"]
    assert "frame reset to about:blank" in result["error"]
    assert post_nav_tree_reads["count"] >= 3
    assert public_frame._data["url"] == "about:blank"
    blank_calls = [
        call
        for call in supervisor.cdp_calls
        if call["method"] == "Page.navigate"
        and call["params"].get("url") == "about:blank"
    ]
    assert len(blank_calls) == 1
    assert [call["method"] for call in supervisor.cdp_calls[:3]] == [
        "Page.enable",
        "Page.getFrameTree",
        "Page.navigate",
    ]


def test_target_reload_blocked_when_selected_target_is_private(cdp_server, monkeypatch):
    """Page.reload must not attach to an already-private page target."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: url != PRIVATE_URL)
    cdp_server.on(
        "Target.getTargets",
        lambda params, sid: {
            "targetInfos": [
                {"targetId": "private-page", "type": "page", "url": PRIVATE_URL}
            ]
        },
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Page.reload",
            target_id="private-page",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "private or internal address" in result["error"]
    assert [call["method"] for call in cdp_server.received()] == ["Target.getTargets"]


def test_frame_id_revalidates_live_url_before_dispatch(monkeypatch):
    """Public snapshot must not win if the live OOPIF navigates private before _cdp."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    # Hermetic runner has no DNS; keep always-blocked IMDS URLs blocked via
    # _is_always_blocked_url while treating ordinary public hosts as safe.
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: True)

    public_frame = _FakeFrame(
        frame_id="oopif-race",
        url="https://example.com/widget",
        origin="https://example.com",
        session_id="child-session",
    )
    private_frame = _FakeFrame(
        frame_id="oopif-race",
        url=PRIVATE_URL,
        origin="http://169.254.169.254",
        session_id="child-session",  # same session — mirrors _on_frame_navigated
    )

    class _RaceSupervisor(_FakeSupervisor):
        def __init__(self):
            super().__init__(
                frame_tree={
                    "top": {
                        "frame_id": "top-1",
                        "url": "https://example.com/",
                        "origin": "https://example.com",
                    },
                    "children": [public_frame.to_dict()],
                },
                frames={"oopif-race": public_frame},
            )
            self._loop = type(
                "Loop",
                (),
                {"is_running": staticmethod(lambda: True)},
            )()

        async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
            self.cdp_calls.append(
                {
                    "method": method,
                    "params": params,
                    "session_id": session_id,
                    "live_url": self._frames["oopif-race"].to_dict().get("url"),
                }
            )
            return {"result": {"ok": True}}

    supervisor = _RaceSupervisor()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    def fake_schedule(coro, loop):
        # Simulate Page.frameNavigated after the copied snapshot check and
        # before the supervisor-loop dispatch body runs.
        supervisor._frames["oopif-race"] = private_frame

        class _Fut:
            def result(self, timeout=None):
                return asyncio.run(coro)

        return _Fut()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="oopif-race",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert "private or internal address" in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_blocks_when_live_frame_missing_before_dispatch(monkeypatch):
    """Fail closed if the selected frame disappears after the snapshot check."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: True)

    public_frame = _FakeFrame(
        frame_id="oopif-gone",
        url="https://example.com/widget",
        origin="https://example.com",
        session_id="child-session",
    )

    class _GoneSupervisor(_FakeSupervisor):
        def __init__(self):
            super().__init__(
                frame_tree={
                    "top": {
                        "frame_id": "top-1",
                        "url": "https://example.com/",
                        "origin": "https://example.com",
                    },
                    "children": [public_frame.to_dict()],
                },
                frames={"oopif-gone": public_frame},
            )
            self._loop = type(
                "Loop",
                (),
                {"is_running": staticmethod(lambda: True)},
            )()

        async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
            self.cdp_calls.append({"method": method, "session_id": session_id})
            return {"result": {"ok": True}}

    supervisor = _GoneSupervisor()
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    def fake_schedule(coro, loop):
        supervisor._frames.pop("oopif-gone", None)

        class _Fut:
            def result(self, timeout=None):
                return asyncio.run(coro)

        return _Fut()

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule
    )

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "1+1"},
            frame_id="oopif-gone",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "missing or transitioning" in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_blocks_oopif_with_empty_url_and_origin(monkeypatch):
    """OOPIF session without URL/origin metadata must fail closed for page CDP."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    supervisor = _FakeSupervisor(
        frame_tree={
            "top": {
                "frame_id": "top-1",
                "url": "https://example.com/",
                "origin": "https://example.com",
            },
            "children": [
                {
                    "frame_id": "oopif-pending",
                    "url": "",
                    "origin": "",
                    "session_id": "child-session",
                }
            ],
        }
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="oopif-pending",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "no URL/origin metadata" in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_fails_closed_when_selected_frame_probe_raises(monkeypatch):
    """Active SSRF guard + probe exception must not fail-open into child CDP."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    def boom(*_args, **_kwargs):
        raise RuntimeError("url safety probe exploded")

    monkeypatch.setattr(
        browser_cdp_tool, "_private_address_from_candidates", boom
    )

    supervisor = _FakeSupervisor(
        frame_tree={
            "top": {
                "frame_id": "top-1",
                "url": "https://example.com/",
                "origin": "https://example.com",
            },
            "children": [
                {
                    "frame_id": "oopif-public",
                    "url": "https://example.com/widget",
                    "origin": "https://example.com",
                    "session_id": "child-session",
                }
            ],
        }
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="oopif-public",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "guard probe failed" in result["error"]
    assert supervisor.cdp_calls == []


def test_frame_id_fails_closed_when_guard_activation_probe_raises(monkeypatch):
    """Activation-probe exception must not fail-open into child CDP."""
    import tools.browser_tool as bt
    import tools.browser_supervisor as bs

    def boom_activation(_task_id):
        raise RuntimeError("ssrf guard activation probe exploded")

    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", boom_activation)
    monkeypatch.setattr(bt, "_current_page_private_url", lambda task_id: None)

    supervisor = _FakeSupervisor(
        frame_tree={
            "top": {
                "frame_id": "top-1",
                "url": "https://example.com/",
                "origin": "https://example.com",
            },
            "children": [
                {
                    "frame_id": "oopif-public",
                    "url": "https://example.com/widget",
                    "origin": "https://example.com",
                    "session_id": "child-session",
                }
            ],
        }
    )
    monkeypatch.setattr(bs.SUPERVISOR_REGISTRY, "get", lambda task_id: supervisor)

    result = json.loads(
        browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="oopif-public",
            task_id="task-1",
        )
    )

    assert "error" in result
    assert "activation probe failed" in result["error"]
    assert supervisor.cdp_calls == []


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
