import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli import mcp_startup
from tui_gateway import server
from tui_gateway import ws as ws_mod


@pytest.fixture(autouse=True)
def _disable_background_mcp_discovery(monkeypatch):
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **kwargs: None,
    )


def test_ws_startup_starts_background_mcp_discovery(monkeypatch):
    """The WS sidecar starts MCP discovery before the first agent build."""
    calls = []
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **kw: calls.append(kw),
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    server._sessions.clear()
    try:
        asyncio.run(ws_mod.handle_ws(FakeWS()))
    finally:
        server._sessions.clear()

    assert calls == [{"logger": ws_mod._log, "thread_name": "tui-ws-mcp-discovery"}]


def test_write_returns_false_when_closed():
    loop = asyncio.new_event_loop()
    transport = ws_mod.WSTransport(AsyncMock(), loop, peer="test-peer")
    try:
        transport._closed = True
        assert transport.write({}) is False
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_write_async_returns_false_when_closed():
    transport = ws_mod.WSTransport(AsyncMock(), asyncio.get_running_loop(), peer="test-peer")
    transport._closed = True
    assert await transport.write_async({}) is False


@pytest.mark.asyncio
async def test_safe_send_success():
    ws = AsyncMock()
    transport = ws_mod.WSTransport(ws, asyncio.get_running_loop(), peer="test-peer")
    await transport._safe_send("test")
    assert transport._closed is False
    ws.send_text.assert_awaited_once_with("test")


@pytest.mark.asyncio
async def test_safe_send_sets_closed_on_exception():
    ws = AsyncMock()
    ws.send_text.side_effect = ConnectionError("gone")
    transport = ws_mod.WSTransport(ws, asyncio.get_running_loop(), peer="test-peer")
    await transport._safe_send("test")
    assert transport._closed is True


@pytest.mark.asyncio
async def test_write_async_success():
    ws = AsyncMock()
    transport = ws_mod.WSTransport(ws, asyncio.get_running_loop(), peer="test-peer")
    assert await transport.write_async({"key": "value"}) is True
    ws.send_text.assert_awaited_once_with('{"key": "value"}')


@pytest.mark.asyncio
async def test_write_batches_streaming_frames_in_order():
    sent = []

    async def send_text(line):
        sent.append(json.loads(line))

    ws = AsyncMock()
    ws.send_text.side_effect = send_text
    transport = ws_mod.WSTransport(ws, asyncio.get_running_loop(), peer="test-peer")
    send_many = AsyncMock(wraps=transport._safe_send_many)
    transport._safe_send_many = send_many
    first = {"params": {"type": "message.delta", "value": "a"}}
    second = {"params": {"type": "message.delta", "value": "b"}}
    response = {"jsonrpc": "2.0", "result": "ok", "id": 1}

    assert transport.write(first) is True
    assert transport.write(second) is True
    assert len(transport._pending_tokens) == 2
    assert ws.send_text.await_count == 0
    assert transport.write(response) is True
    await asyncio.sleep(0)

    send_many.assert_awaited_once_with(
        [json.dumps(first), json.dumps(second), json.dumps(response)]
    )
    assert sent == [first, second, response]


@pytest.mark.asyncio
async def test_safe_send_many_stops_after_failure():
    ws = AsyncMock()
    ws.send_text.side_effect = [None, ConnectionError("gone"), None]
    transport = ws_mod.WSTransport(ws, asyncio.get_running_loop(), peer="test-peer")

    await transport._safe_send_many(["first", "second", "third"])

    assert transport._closed is True
    assert ws.send_text.await_count == 2


@pytest.mark.asyncio
async def test_close_cancels_token_flush_timer():
    transport = ws_mod.WSTransport(
        AsyncMock(), asyncio.get_running_loop(), peer="test-peer"
    )
    assert transport.write({"params": {"type": "message.delta"}}) is True
    await asyncio.sleep(0)
    handle = transport._token_flush_handle
    assert handle is not None

    transport.close()

    assert handle.cancelled() is True
    assert transport._token_flush_handle is None


def test_peer_label_host_and_port():
    ws = MagicMock()
    ws.client.host = "1.2.3.4"
    ws.client.port = 5678
    assert ws_mod._ws_peer_label(ws) == "1.2.3.4:5678"


def test_peer_label_host_no_port():
    ws = MagicMock()
    ws.client.host = "1.2.3.4"
    ws.client.port = None
    assert ws_mod._ws_peer_label(ws) == "1.2.3.4"


def test_peer_label_no_client():
    assert ws_mod._ws_peer_label(MagicMock(spec=[])) == "unknown"


def _run_disconnect(monkeypatch, seed):
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    def _fake_finalize(s, end_reason="tui_close"):
        worker = s.get("slash_worker")
        if worker:
            worker.close()

    monkeypatch.setattr(server, "_finalize_session", _fake_finalize)

    created = []
    real_transport = ws_mod.WSTransport
    monkeypatch.setattr(
        ws_mod,
        "WSTransport",
        lambda ws, loop, **kw: created.append(real_transport(ws, loop, **kw))
        or created[-1],
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            seed(created[0])
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))


def test_ws_disconnect_reaps_flagged_session_and_closes_worker(monkeypatch):
    closed = []

    class FakeWorker:
        def close(self):
            closed.append(True)

    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda transport: server._sessions.update(
                flagged={
                    "transport": transport,
                    "close_on_disconnect": True,
                    "slash_worker": FakeWorker(),
                    "session_key": "k",
                }
            ),
        )
        assert "flagged" not in server._sessions
        assert closed == [True]
    finally:
        server._sessions.clear()


def test_ws_disconnect_preserves_and_repoints_reconnectable_session(monkeypatch):
    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda transport: server._sessions.update(
                plain={
                    "transport": transport,
                    "close_on_disconnect": False,
                    "session_key": "k",
                }
            ),
        )
        assert server._sessions["plain"]["transport"] is server._detached_ws_transport
    finally:
        server._sessions.clear()


def test_ws_write_loop_stall_does_not_latch_transport(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_WRITE_TIMEOUT_S", 0.05)
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(line)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="stall-test")
        loop.call_soon_threadsafe(time.sleep, 0.3)
        assert transport.write({"a": 1}) is True
        assert transport._closed is False
        assert transport.write({"b": 2}) is True
        deadline = time.time() + 2
        while len(sent) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(sent) == 2
        assert transport._closed is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_ready_send_fails(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock(host="127.0.0.1", port=9999)
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()
    ws.send_text.side_effect = ConnectionError("refused")

    await ws_mod.handle_ws(ws)

    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_parse_error(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock(host="127.0.0.1", port=9999)
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()
    received = iter(["not-valid-json{{"])

    async def receive_text():
        try:
            return next(received)
        except StopIteration:
            raise ws_mod._WebSocketDisconnect()

    ws.receive_text.side_effect = receive_text

    await ws_mod.handle_ws(ws)

    ws.accept.assert_awaited_once()
    assert ws.send_text.await_count == 2
    error_payload = json.loads(ws.send_text.await_args_list[1][0][0])
    assert error_payload["error"]["code"] == -32700


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_dispatch_success(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock(host="127.0.0.1", port=9999)
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()
    rpc_request = {"jsonrpc": "2.0", "method": "test.echo", "id": 1}
    rpc_response = {"jsonrpc": "2.0", "result": "ok", "id": 1}
    ws.receive_text.side_effect = [json.dumps(rpc_request), ws_mod._WebSocketDisconnect()]
    mock_server.dispatch = MagicMock(return_value=rpc_response)

    await ws_mod.handle_ws(ws)

    ws.accept.assert_awaited_once()
    mock_server.dispatch.assert_called_once()
    assert ws.send_text.await_count == 2
    response_payload = json.loads(ws.send_text.await_args_list[1][0][0])
    assert response_payload["result"] == "ok"
    assert response_payload["id"] == 1
