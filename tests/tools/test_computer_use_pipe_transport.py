"""Unit tests for NativePipeComputerUseTransport and cua-driver named pipe IPC."""

import base64
import json
import os
import struct
import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use.cua_backend_pipe import (
    NativePipeComputerUseTransport,
    decode_message_frame,
    encode_message_frame,
    get_computer_use_pipe_path,
    is_named_pipe_available,
)
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession, _orig_subprocess_run


def test_frame_encoding_and_decoding_le():
    payload = b'{"method": "ping"}'
    framed = encode_message_frame(payload, "le")
    assert len(framed) == 4 + len(payload)
    length = struct.unpack("<I", framed[:4])[0]
    assert length == len(payload)

    msgs, remainder = decode_message_frame(framed, "le")
    assert msgs == [payload]
    assert remainder == b""


def test_frame_encoding_and_decoding_be():
    payload = b'{"method": "ping"}'
    framed = encode_message_frame(payload, "be")
    assert len(framed) == 4 + len(payload)
    length = struct.unpack(">I", framed[:4])[0]
    assert length == len(payload)

    msgs, remainder = decode_message_frame(framed, "be")
    assert msgs == [payload]
    assert remainder == b""


def test_decode_message_frame_fragmented_and_multiple():
    msg1 = b"hello"
    msg2 = b"world123"
    framed = encode_message_frame(msg1, "le") + encode_message_frame(msg2, "le")

    # Partial read (missing part of second body)
    partial_bytes = framed[:-3]
    msgs, rem = decode_message_frame(partial_bytes, "le")
    assert msgs == [msg1]
    assert rem == encode_message_frame(msg2, "le")[:-3]

    # Full read
    msgs_all, rem_all = decode_message_frame(framed, "le")
    assert msgs_all == [msg1, msg2]
    assert rem_all == b""


def test_get_computer_use_pipe_path_env(monkeypatch):
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    monkeypatch.setenv("SKY_CUA_NATIVE_PIPE_DIRECTORY", r"\\.\pipe\custom-zcode-pipe")
    assert get_computer_use_pipe_path() == r"\\.\pipe\custom-zcode-pipe"

    monkeypatch.delenv("SKY_CUA_NATIVE_PIPE_DIRECTORY", raising=False)
    monkeypatch.setenv("CUA_DRIVER_PIPE_PATH", "my_driver_pipe")
    assert get_computer_use_pipe_path() == r"\\.\pipe\my_driver_pipe"

    monkeypatch.delenv("CUA_DRIVER_PIPE_PATH", raising=False)
    monkeypatch.setenv("CUA_DRIVER_SOCKET", r"\\.\pipe\test-sock")
    assert get_computer_use_pipe_path() == r"\\.\pipe\test-sock"

    monkeypatch.delenv("CUA_DRIVER_SOCKET", raising=False)
    assert get_computer_use_pipe_path() == r"\\.\pipe\cua-driver"


def test_get_computer_use_pipe_path_embedded_daemon(monkeypatch):
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    daemon = MagicMock(socket_path=r"\\.\pipe\daemon-pipe-123")
    assert get_computer_use_pipe_path(embedded_daemon=daemon) == r"\\.\pipe\daemon-pipe-123"


def test_is_named_pipe_available_win32(monkeypatch):
    if sys.platform != "win32":
        # On non-win32 it returns False immediately
        assert is_named_pipe_available() is False
        return

    # Mock kernel32.WaitNamedPipeW returning 1 (pipe listening)
    with patch("ctypes.windll.kernel32.WaitNamedPipeW", return_value=1):
        assert is_named_pipe_available(r"\\.\pipe\test") is True

    # Mock WaitNamedPipeW returning 0 and GetLastError() returning 231 (ERROR_PIPE_BUSY)
    with patch("ctypes.windll.kernel32.WaitNamedPipeW", return_value=0), \
         patch("ctypes.GetLastError", return_value=231):
        assert is_named_pipe_available(r"\\.\pipe\test") is True

    # Mock WaitNamedPipeW returning 0 and GetLastError() returning 2 (ERROR_FILE_NOT_FOUND)
    with patch("ctypes.windll.kernel32.WaitNamedPipeW", return_value=0), \
         patch("ctypes.GetLastError", return_value=2):
        assert is_named_pipe_available(r"\\.\pipe\test") is False


def test_normalize_response_success(monkeypatch):
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    transport = NativePipeComputerUseTransport(pipe_path=r"\\.\pipe\fake")
    resp = {
        "ok": True,
        "result": {
            "content": [
                {"type": "text", "text": "Clicked successfully"},
                {"type": "image", "data": "base64imagebytes"}
            ],
            "structuredContent": {"verified": True, "effect": "confirmed"},
            "isError": False
        }
    }
    norm = transport._normalize_response(resp, "click", None)
    assert norm["data"] == "Clicked successfully"
    assert norm["images"] == ["base64imagebytes"]
    assert norm["structuredContent"] == {"verified": True, "effect": "confirmed"}
    assert norm["isError"] is False


def test_normalize_response_error(monkeypatch):
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    transport = NativePipeComputerUseTransport(pipe_path=r"\\.\pipe\fake")
    resp = {
        "ok": False,
        "error": "Failed to locate window",
        "exit_code": 2
    }
    norm = transport._normalize_response(resp, "click", None)
    assert norm["isError"] is True
    assert "Failed to locate window" in norm["data"]
    assert norm["structuredContent"]["exit_code"] == 2


def test_session_cli_fallback_uses_pipe_when_available(monkeypatch):
    """When on Windows with unpatched subprocess.run and pipe available, _call_tool_via_cli uses pipe."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    session = _CuaDriverSession(_AsyncBridge())
    mock_pipe = MagicMock()
    mock_pipe.call_tool.return_value = {"data": "from-pipe", "images": [], "structuredContent": {}, "isError": False}

    monkeypatch.setattr(session, "_get_pipe_transport", lambda: mock_pipe)
    res = session._call_tool_via_cli("list_windows", {}, timeout=5.0)
    assert res["data"] == "from-pipe"
    mock_pipe.call_tool.assert_called_once()


def test_session_cli_fallback_bypasses_pipe_when_disabled(monkeypatch):
    """When HERMES_CUA_DISABLE_PIPE is set, _call_tool_via_cli falls back to CLI subprocess."""
    session = _CuaDriverSession(_AsyncBridge())
    monkeypatch.setenv("HERMES_CUA_DISABLE_PIPE", "1")

    proc = MagicMock(stdout='{"data": "from-cli"}', stderr="", returncode=0)
    with patch("tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd", return_value="cua-driver"), \
         patch("subprocess.run", return_value=proc):
        res = session._call_tool_via_cli("list_windows", {}, timeout=5.0)
        assert res["structuredContent"] == {"data": "from-cli"}


def test_session_cli_fallback_recovers_if_pipe_raises(monkeypatch):
    """If pipe transport throws an unexpected exception, fall back to subprocess."""
    session = _CuaDriverSession(_AsyncBridge())
    mock_pipe = MagicMock()
    mock_pipe.call_tool.side_effect = RuntimeError("Broken pipe")

    monkeypatch.setattr(session, "_get_pipe_transport", lambda: mock_pipe)
    proc = MagicMock(stdout='{"data": "from-subprocess"}', stderr="", returncode=0)
    with patch("tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd", return_value="cua-driver"), \
         patch("subprocess.run", return_value=proc):
        res = session._call_tool_via_cli("list_windows", {}, timeout=5.0)
        assert res["structuredContent"] == {"data": "from-subprocess"}


def test_normalize_response_jsonrpc_error(monkeypatch):
    """Ensure standard JSON-RPC 2.0 error dictionaries are classified as isError: True."""
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    transport = NativePipeComputerUseTransport(pipe_path=r"\\.\pipe\fake")
    resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32000,
            "message": "Access denied by security policy",
        }
    }
    norm = transport._normalize_response(resp, "click", None)
    assert norm["isError"] is True
    assert "Access denied" in norm["data"]
    assert norm["structuredContent"]["exit_code"] == -32000
    assert norm["structuredContent"]["ok"] is False


def test_session_id_none_omits_session_id_in_payload(monkeypatch):
    """When session_id is None, call_tool must not inject a stale 'hermes' session id."""
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    transport = NativePipeComputerUseTransport(pipe_path=r"\\.\pipe\fake", session_id=None)
    assert transport.session_id is None

    # Verify that when session_id is None and call_args has no session, payload omits session_id
    mock_handle = MagicMock()
    written_data = []

    def mock_write(h, data):
        written_data.append(data)

    monkeypatch.setattr(transport, "_open_handle", lambda timeout_sec: mock_handle)
    monkeypatch.setattr(transport, "_read_all_line", lambda h, deadline: b'{"ok": true, "result": {}}\n')
    with patch("_winapi.WriteFile", side_effect=mock_write), \
         patch("_winapi.CloseHandle"):
        transport.call_tool("list_windows", {})

    assert len(written_data) == 1
    sent_json = json.loads(written_data[0].decode("utf-8").strip())
    assert "session_id" not in sent_json


def test_read_all_line_times_out(monkeypatch):
    """Ensure _read_all_line raises TimeoutError if deadline expires."""
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")

    transport = NativePipeComputerUseTransport(pipe_path=r"\\.\pipe\fake")
    import ctypes

    # Mock PeekNamedPipe returning 0 bytes available
    def mock_peek(handle, buf, size, bread, bavail, bleft):
        if bavail:
            ctypes.cast(bavail, ctypes.POINTER(ctypes.c_ulong)).contents.value = 0
        return 1

    with patch("ctypes.windll.kernel32.PeekNamedPipe", side_effect=mock_peek):
        with pytest.raises(TimeoutError):
            transport._read_all_line(MagicMock(), deadline=0.0)


def test_call_tool_healthy_windows_chooses_pipe_directly(monkeypatch):
    """(a) On Windows, healthy requests in call_tool() take the direct named pipe transport fast path."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    session = _CuaDriverSession(_AsyncBridge())
    session._started = True

    mock_pipe = MagicMock()
    mock_pipe.call_tool.return_value = {
        "data": "from-pipe",
        "images": [],
        "structuredContent": {"success": True},
        "isError": False,
    }
    monkeypatch.setattr(session, "_get_pipe_transport", lambda: mock_pipe)

    mock_mcp = MagicMock()
    monkeypatch.setattr(session._bridge, "run", mock_mcp)

    res = session.call_tool("click", {"element": 5}, timeout=10.0)
    assert res["data"] == "from-pipe"
    mock_pipe.call_tool.assert_called_once_with("click", {"element": 5}, timeout=10.0)
    mock_mcp.assert_not_called()


def test_call_tool_unavailable_or_pre_send_pipe_falls_back_to_mcp(monkeypatch):
    """(b) When pipe is unavailable or fails before sending bytes (or for replay-safe tools), next transport (MCP) is chosen."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    from tools.computer_use.cua_backend_pipe import PipePreDispatchError

    # Sub-case 1: Pipe is unavailable (_get_pipe_transport returns None)
    session1 = _CuaDriverSession(_AsyncBridge())
    session1._started = True
    monkeypatch.setattr(session1, "_get_pipe_transport", lambda: None)
    mock_mcp1 = MagicMock(return_value={"data": "from-mcp-1", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session1._bridge, "run", mock_mcp1)

    res1 = session1.call_tool("click", {"element": 1}, timeout=5.0)
    assert res1["data"] == "from-mcp-1"
    mock_mcp1.assert_called_once()

    # Sub-case 2: Pipe connect fails pre-dispatch (proven pre-send)
    session2 = _CuaDriverSession(_AsyncBridge())
    session2._started = True
    mock_pipe2 = MagicMock()
    mock_pipe2.call_tool.side_effect = PipePreDispatchError("Failed to connect to pipe")
    monkeypatch.setattr(session2, "_get_pipe_transport", lambda: mock_pipe2)
    mock_mcp2 = MagicMock(return_value={"data": "from-mcp-2", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session2._bridge, "run", mock_mcp2)

    res2 = session2.call_tool("click", {"element": 2}, timeout=5.0)
    assert res2["data"] == "from-mcp-2"
    mock_pipe2.call_tool.assert_called_once()
    mock_mcp2.assert_called_once()

    # Sub-case 3: Tool is in _TRANSPORT_REPLAY_SAFE_TOOLS (e.g. list_windows) and pipe fails
    session3 = _CuaDriverSession(_AsyncBridge())
    session3._started = True
    mock_pipe3 = MagicMock()
    mock_pipe3.call_tool.side_effect = RuntimeError("Broken pipe stream")
    monkeypatch.setattr(session3, "_get_pipe_transport", lambda: mock_pipe3)
    mock_mcp3 = MagicMock(return_value={"data": "windows-mcp", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session3._bridge, "run", mock_mcp3)

    res3 = session3.call_tool("list_windows", {}, timeout=5.0)
    assert res3["data"] == "windows-mcp"
    mock_pipe3.call_tool.assert_called_once()
    mock_mcp3.assert_called_once()


def test_call_tool_post_dispatch_failure_on_mutation_fails_closed_without_replay(monkeypatch):
    """(c) Post-dispatch timeout or broken response on a mutation yields *_outcome_unknown and is NOT replayed."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    from tools.computer_use.cua_backend_pipe import PipePostDispatchError, PipePostDispatchTimeoutError

    # Sub-case 1: Post-dispatch timeout yields timeout_outcome_unknown and does NOT call MCP
    session1 = _CuaDriverSession(_AsyncBridge())
    session1._started = True
    mock_pipe1 = MagicMock()
    mock_pipe1.call_tool.side_effect = PipePostDispatchTimeoutError("Pipe response timed out after bytes were sent")
    monkeypatch.setattr(session1, "_get_pipe_transport", lambda: mock_pipe1)
    mock_mcp1 = MagicMock()
    monkeypatch.setattr(session1._bridge, "run", mock_mcp1)

    res1 = session1.call_tool("click", {"element": 10}, timeout=5.0)
    assert res1["isError"] is True
    assert res1["structuredContent"]["code"] == "timeout_outcome_unknown"
    mock_pipe1.call_tool.assert_called_once()
    mock_mcp1.assert_not_called()
    assert session1._timeout_suspect is True

    # Sub-case 2: Generic TimeoutError on mutation also yields timeout_outcome_unknown without replay
    session2 = _CuaDriverSession(_AsyncBridge())
    session2._started = True
    mock_pipe2 = MagicMock()
    mock_pipe2.call_tool.side_effect = TimeoutError("Deadline reached waiting for pipe")
    monkeypatch.setattr(session2, "_get_pipe_transport", lambda: mock_pipe2)
    mock_mcp2 = MagicMock()
    monkeypatch.setattr(session2._bridge, "run", mock_mcp2)

    res2 = session2.call_tool("type", {"text": "hello"}, timeout=5.0)
    assert res2["isError"] is True
    assert res2["structuredContent"]["code"] == "timeout_outcome_unknown"
    mock_pipe2.call_tool.assert_called_once()
    mock_mcp2.assert_not_called()

    # Sub-case 3: Post-dispatch transport error (broken pipe / empty response) yields transport_outcome_unknown without replay
    session3 = _CuaDriverSession(_AsyncBridge())
    session3._started = True
    mock_pipe3 = MagicMock()
    mock_pipe3.call_tool.side_effect = PipePostDispatchError("cua-driver named pipe returned empty response for tool click")
    monkeypatch.setattr(session3, "_get_pipe_transport", lambda: mock_pipe3)
    mock_mcp3 = MagicMock()
    monkeypatch.setattr(session3._bridge, "run", mock_mcp3)

    res3 = session3.call_tool("click", {"element": 12}, timeout=5.0)
    assert res3["isError"] is True
    assert res3["structuredContent"]["code"] == "transport_outcome_unknown"
    mock_pipe3.call_tool.assert_called_once()
    mock_mcp3.assert_not_called()


def test_lifecycle_start_and_end_session_routes_over_pipe(monkeypatch):
    """start_session and end_session route over named pipe on Windows when pipe is available."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    session = _CuaDriverSession(_AsyncBridge())
    session._started = True

    mock_pipe = MagicMock()
    mock_pipe.call_tool.return_value = {
        "data": "session-started",
        "images": [],
        "structuredContent": {"session": "test-sess-1"},
        "isError": False,
    }
    monkeypatch.setattr(session, "_get_pipe_transport", lambda: mock_pipe)

    mock_mcp = MagicMock()
    monkeypatch.setattr(session._bridge, "run", mock_mcp)

    # 1. start_session routed to pipe
    res = session.call_tool("start_session", {"session": "test-sess-1"})
    assert res["data"] == "session-started"
    mock_pipe.call_tool.assert_called_once_with("start_session", {"session": "test-sess-1"}, timeout=30.0)
    mock_mcp.assert_not_called()
    assert session._declared_session_id == "test-sess-1"
    assert mock_pipe.session_id == "test-sess-1"

    # 2. end_session routed to pipe
    mock_pipe.call_tool.reset_mock()
    mock_pipe.call_tool.return_value = {
        "data": "session-ended",
        "images": [],
        "structuredContent": {},
        "isError": False,
    }
    res_end = session.call_tool("end_session", {"session": "test-sess-1"})
    assert res_end["data"] == "session-ended"
    mock_pipe.call_tool.assert_called_once_with("end_session", {"session": "test-sess-1"}, timeout=30.0)
    mock_mcp.assert_not_called()
    assert session._declared_session_id is None
    assert mock_pipe.session_id is None


def test_session_not_available_pipe_error_falls_back_to_mcp_for_replay_safe_tools(monkeypatch):
    """'session is not available to this transport' resets pipe transport and falls back to MCP for replay-safe tools."""
    if sys.platform != "win32":
        pytest.skip("Windows pipe test")

    from tools.computer_use.cua_backend_pipe import PipePostDispatchError

    # Sub-case 1: Pipe response carries isError: True with 'session is not available to this transport' on list_windows
    session1 = _CuaDriverSession(_AsyncBridge())
    session1._started = True
    mock_pipe1 = MagicMock()
    mock_pipe1.call_tool.return_value = {
        "isError": True,
        "data": "session is not available to this transport",
        "images": [],
        "structuredContent": {"error": "session is not available to this transport"},
    }
    session1._pipe_transport = mock_pipe1
    monkeypatch.setattr(session1, "_get_pipe_transport", lambda: mock_pipe1)
    mock_mcp1 = MagicMock(return_value={"data": "windows-from-mcp", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session1._bridge, "run", mock_mcp1)

    res1 = session1.call_tool("list_windows", {"session": "sess-abc"}, timeout=5.0)
    assert res1["data"] == "windows-from-mcp"
    assert session1._pipe_transport is None
    mock_pipe1.call_tool.assert_called_once()
    mock_mcp1.assert_called_once()

    # Sub-case 2: Pipe raises PipePostDispatchError with 'session is not available to this transport' on list_apps
    session2 = _CuaDriverSession(_AsyncBridge())
    session2._started = True
    mock_pipe2 = MagicMock()
    mock_pipe2.call_tool.side_effect = PipePostDispatchError("cua-driver error: session is not available to this transport")
    session2._pipe_transport = mock_pipe2
    monkeypatch.setattr(session2, "_get_pipe_transport", lambda: mock_pipe2)
    mock_mcp2 = MagicMock(return_value={"data": "apps-from-mcp", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session2._bridge, "run", mock_mcp2)

    res2 = session2.call_tool("list_apps", {"session": "sess-abc"}, timeout=5.0)
    assert res2["data"] == "apps-from-mcp"
    assert session2._pipe_transport is None
    mock_pipe2.call_tool.assert_called_once()
    mock_mcp2.assert_called_once()

    # Sub-case 3: Mutating tool (click) receiving 'session is not available to this transport' fails closed WITHOUT replay
    session3 = _CuaDriverSession(_AsyncBridge())
    session3._started = True
    mock_pipe3 = MagicMock()
    mock_pipe3.call_tool.return_value = {
        "isError": True,
        "data": "session is not available to this transport",
        "images": [],
        "structuredContent": {"error": "session is not available to this transport"},
    }
    session3._pipe_transport = mock_pipe3
    monkeypatch.setattr(session3, "_get_pipe_transport", lambda: mock_pipe3)
    mock_mcp3 = MagicMock()
    monkeypatch.setattr(session3._bridge, "run", mock_mcp3)

    res3 = session3.call_tool("click", {"element": 1}, timeout=5.0)
    assert res3["isError"] is True
    assert "session is not available to this transport" in res3["data"]
    assert session3._pipe_transport is None
    mock_pipe3.call_tool.assert_called_once()
    mock_mcp3.assert_not_called()

    # Sub-case 4: If a session had a transport mismatch on list_windows, subsequent mutating calls (click)
    # route over MCP stdio where the session lives instead of attempting pipe and failing closed.
    session4 = _CuaDriverSession(_AsyncBridge())
    session4._started = True
    session4._declared_session_id = "sess-pinned"
    mock_pipe4 = MagicMock()
    mock_pipe4.call_tool.return_value = {
        "isError": True,
        "data": "session is not available to this transport",
        "images": [],
        "structuredContent": {"error": "session is not available to this transport"},
    }
    session4._pipe_transport = mock_pipe4
    # Simulate list_windows encountering transport mismatch and pinning session to MCP
    mock_mcp4 = MagicMock(return_value={"data": "win-ok", "images": [], "structuredContent": {}, "isError": False})
    monkeypatch.setattr(session4._bridge, "run", mock_mcp4)
    res_list = session4.call_tool("list_windows", {"session": "sess-pinned"}, timeout=5.0)
    assert res_list["data"] == "win-ok"
    assert session4._pipe_disabled_for_session == "sess-pinned"

    # Now click must route to MCP stdio, NOT fail on pipe
    mock_pipe4.call_tool.reset_mock()
    mock_mcp4.reset_mock()
    mock_mcp4.return_value = {"data": "click-ok", "images": [], "structuredContent": {}, "isError": False}
    res_click = session4.call_tool("click", {"element": 1}, timeout=5.0)
    assert res_click["data"] == "click-ok"
    mock_pipe4.call_tool.assert_not_called()
    mock_mcp4.assert_called_once()


def test_list_apps_propagates_transport_error():
    """list_apps() uses _call_capture_tool so transport/logical errors raise RuntimeError instead of returning []."""
    from tools.computer_use.cua_backend_capture import _CaptureMixin

    class CaptureTestBackend(_CaptureMixin):
        def __init__(self, session):
            self._session = session
            self._session_id = "sess-test"
            self._active_target = None
            self._active_pid = None
            self._active_window_id = None
            self._last_app = None

        def _clear_active_target(self):
            self._active_target = None
            self._active_pid = None
            self._active_window_id = None

    # Error case: transport error must raise RuntimeError
    mock_session = MagicMock()
    mock_session.call_tool.return_value = {
        "isError": True,
        "data": "session is not available to this transport",
        "images": [],
        "structuredContent": None,
    }
    backend = CaptureTestBackend(mock_session)
    with pytest.raises(RuntimeError) as exc_info:
        backend.list_apps()
    assert "cua-driver list_apps failed: session is not available to this transport" in str(exc_info.value)

    # Success case: valid apps parsed properly
    mock_session.call_tool.return_value = {
        "isError": False,
        "data": None,
        "images": [],
        "structuredContent": {"apps": [{"name": "calculator.exe", "pid": 1234}]},
    }
    apps = backend.list_apps()
    assert apps == [{"name": "calculator.exe", "pid": 1234}]



