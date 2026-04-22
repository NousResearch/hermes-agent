"""Tests for MCP tool-handler reconnect recovery paths."""

import json
import threading
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def test_is_session_identity_error_detects_known_markers():
    from tools.mcp_tool import _is_session_identity_error

    assert _is_session_identity_error(RuntimeError("Invalid params: Invalid or expired session")) is True
    assert _is_session_identity_error(RuntimeError("Session expired")) is True
    assert _is_session_identity_error(RuntimeError("session not found")) is True
    assert _is_session_identity_error(RuntimeError("Unknown session: abc123")) is True
    assert _is_session_identity_error(RuntimeError("invalid or missing session id")) is True



def test_is_transport_dead_error_detects_known_markers():
    from tools.mcp_tool import _is_transport_dead_error

    assert _is_transport_dead_error(RuntimeError("Session terminated")) is True
    assert _is_transport_dead_error(RuntimeError("stream closed by peer")) is True
    assert _is_transport_dead_error(RuntimeError("EndOfStream")) is True
    assert _is_transport_dead_error(RuntimeError("connection closed")) is True
    assert _is_transport_dead_error(RuntimeError("connection reset by peer")) is True
    assert _is_transport_dead_error(RuntimeError("server disconnected unexpectedly")) is True



def test_reconnectable_detectors_reject_unrelated_errors():
    from tools.mcp_tool import _is_session_identity_error, _is_transport_dead_error

    assert _is_session_identity_error(RuntimeError("Tool failed to execute")) is False
    assert _is_transport_dead_error(RuntimeError("Tool failed to execute")) is False
    assert _is_session_identity_error(RuntimeError("401 Unauthorized")) is False
    assert _is_transport_dead_error(RuntimeError("401 Unauthorized")) is False
    assert _is_session_identity_error(InterruptedError("session expired")) is False
    assert _is_transport_dead_error(InterruptedError("session terminated")) is False



def test_classify_recoverable_error_returns_session_identity_error():
    from tools.mcp_tool import _classify_recoverable_error

    kind = _classify_recoverable_error(RuntimeError("invalid or missing session id"))
    assert kind == "session_identity_error"



def test_classify_recoverable_error_returns_transport_dead():
    from tools.mcp_tool import _classify_recoverable_error

    kind = _classify_recoverable_error(RuntimeError("Session terminated"))
    assert kind == "transport_dead"



def test_classify_recoverable_error_returns_none_for_unrelated_error():
    from tools.mcp_tool import _classify_recoverable_error

    assert _classify_recoverable_error(RuntimeError("boom")) is None


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _install_stub_server(name: str = "wpcom"):
    from tools import mcp_tool

    mcp_tool._ensure_mcp_loop()

    server = MagicMock()
    server.name = name
    reconnect_flag = threading.Event()

    class _EventAdapter:
        def set(self):
            reconnect_flag.set()

    server._reconnect_event = _EventAdapter()

    ready_flag = threading.Event()
    ready_flag.set()
    server._ready = MagicMock()
    server._ready.is_set = ready_flag.is_set
    server.session = MagicMock()
    return server, reconnect_flag


# ---------------------------------------------------------------------------
# Dispatcher / reconnect helper coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message, expected_kind",
    [
        ("Invalid or expired session", "session_identity_error"),
        ("invalid or missing session id", "session_identity_error"),
        ("Session terminated", "transport_dead"),
        ("stream closed", "transport_dead"),
    ],
)
def test_handle_recoverable_error_and_retry_dispatches_reconnectable_errors(
    monkeypatch, tmp_path, message, expected_kind
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _classify_recoverable_error, _handle_recoverable_error_and_retry

    server, reconnect_flag = _install_stub_server(f"srv-{expected_kind}")
    mcp_tool._servers[f"srv-{expected_kind}"] = server

    calls = {"n": 0}

    def _retry():
        calls["n"] += 1
        return '{"result": "ok"}'

    try:
        out = _handle_recoverable_error_and_retry(
            f"srv-{expected_kind}",
            RuntimeError(message),
            _retry,
            "tools/call",
        )
        assert _classify_recoverable_error(RuntimeError(message)) == expected_kind
        assert out == '{"result": "ok"}'
        # The successful retry proves the dispatcher selected a reconnectable
        # path; with the background loop involved, the event signal itself does
        # not have to flip synchronously in this unit test.
        assert calls["n"] == 1
    finally:
        mcp_tool._servers.pop(f"srv-{expected_kind}", None)
        mcp_tool._server_error_counts.pop(f"srv-{expected_kind}", None)



def test_handle_recoverable_error_and_retry_returns_none_for_unrelated_error():
    from tools.mcp_tool import _handle_recoverable_error_and_retry

    out = _handle_recoverable_error_and_retry(
        "missing",
        RuntimeError("boom"),
        lambda: '{"result": "ok"}',
        "tools/call",
    )
    assert out is None


@pytest.mark.parametrize(
    "message",
    ["Invalid or expired session", "Session terminated"],
)
def test_reconnectable_handler_returns_none_without_loop(monkeypatch, message):
    from tools import mcp_tool
    from tools.mcp_tool import _handle_reconnectable_error_and_retry

    server = MagicMock()
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set = MagicMock(return_value=True)
    server.session = MagicMock()
    mcp_tool._servers["srv-noloop"] = server

    monkeypatch.setattr(mcp_tool, "_mcp_loop", None)

    try:
        out = _handle_reconnectable_error_and_retry(
            "srv-noloop",
            RuntimeError(message),
            lambda: '{"ok": true}',
            "tools/call",
        )
        assert out is None
    finally:
        mcp_tool._servers.pop("srv-noloop", None)



def test_reconnectable_handler_returns_none_without_server_record():
    from tools.mcp_tool import _handle_reconnectable_error_and_retry

    out = _handle_reconnectable_error_and_retry(
        "does-not-exist",
        RuntimeError("Session terminated"),
        lambda: '{"ok": true}',
        "tools/call",
    )
    assert out is None


@pytest.mark.parametrize(
    "message",
    ["Invalid or expired session", "Session terminated"],
)
def test_reconnectable_handler_returns_none_when_retry_also_fails(monkeypatch, tmp_path, message):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _handle_reconnectable_error_and_retry

    server, _ = _install_stub_server("srv-retry-fail")
    mcp_tool._servers["srv-retry-fail"] = server

    def _retry_raises():
        raise RuntimeError("retry blew up too")

    try:
        out = _handle_reconnectable_error_and_retry(
            "srv-retry-fail",
            RuntimeError(message),
            _retry_raises,
            "tools/call",
        )
        assert out is None
    finally:
        mcp_tool._servers.pop("srv-retry-fail", None)


# ---------------------------------------------------------------------------
# Handler integration — tool call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["Invalid params: Invalid or expired session", "Session terminated"],
)
def test_call_tool_handler_reconnects_on_reconnectable_errors(monkeypatch, tmp_path, message):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    server, reconnect_flag = _install_stub_server("wpcom")
    mcp_tool._servers["wpcom"] = server
    mcp_tool._server_error_counts.pop("wpcom", None)

    call_count = {"n": 0}

    async def _call_sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError(message)
        result = MagicMock()
        result.isError = False
        result.content = [MagicMock(type="text", text="tool completed")]
        result.structuredContent = None
        return result

    server.session.call_tool = _call_sequence

    try:
        handler = _make_tool_handler("wpcom", "wpcom-mcp-content-authoring", 10.0)
        parsed = json.loads(handler({"slug": "hello"}))
        assert "error" not in parsed
        assert parsed["result"] == "tool completed"
        assert reconnect_flag.is_set()
        assert call_count["n"] == 2
    finally:
        mcp_tool._servers.pop("wpcom", None)
        mcp_tool._server_error_counts.pop("wpcom", None)



def test_call_tool_handler_non_reconnectable_error_falls_through(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    server, reconnect_flag = _install_stub_server("srv")
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)

    async def _raises(*a, **kw):
        raise RuntimeError("Tool execution failed — unrelated error")

    server.session.call_tool = _raises

    try:
        handler = _make_tool_handler("srv", "mytool", 10.0)
        parsed = json.loads(handler({"arg": "v"}))
        assert "MCP call failed" in parsed.get("error", "")
        assert not reconnect_flag.is_set()
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


# ---------------------------------------------------------------------------
# Handler integration — resources / prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_factory, handler_kwargs, session_method, op_label",
    [
        ("_make_list_resources_handler", {"tool_timeout": 10.0}, "list_resources", "list_resources"),
        ("_make_read_resource_handler", {"tool_timeout": 10.0}, "read_resource", "read_resource"),
        ("_make_list_prompts_handler", {"tool_timeout": 10.0}, "list_prompts", "list_prompts"),
        ("_make_get_prompt_handler", {"tool_timeout": 10.0}, "get_prompt", "get_prompt"),
    ],
)
@pytest.mark.parametrize("message", ["Invalid or expired session", "Session terminated"])
def test_non_tool_handlers_also_reconnect_on_reconnectable_errors(
    monkeypatch, tmp_path, handler_factory, handler_kwargs, session_method, op_label, message
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    server, reconnect_flag = _install_stub_server(f"srv-{op_label}")
    mcp_tool._servers[f"srv-{op_label}"] = server
    mcp_tool._server_error_counts.pop(f"srv-{op_label}", None)

    call_count = {"n": 0}

    async def _sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError(message)
        result = MagicMock()
        result.resources = []
        result.prompts = []
        result.contents = []
        result.messages = []
        result.description = None
        return result

    setattr(server.session, session_method, _sequence)

    factory = getattr(mcp_tool, handler_factory)
    try:
        handler = factory(f"srv-{op_label}", **handler_kwargs)
        if op_label == "read_resource":
            parsed = json.loads(handler({"uri": "file://foo"}))
        elif op_label == "get_prompt":
            parsed = json.loads(handler({"name": "p1"}))
        else:
            parsed = json.loads(handler({}))
        assert "error" not in parsed
        assert reconnect_flag.is_set()
        assert call_count["n"] == 2
    finally:
        mcp_tool._servers.pop(f"srv-{op_label}", None)
        mcp_tool._server_error_counts.pop(f"srv-{op_label}", None)
