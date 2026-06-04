"""Tests for tui_gateway.ws WebSocket transport layer."""

import asyncio
import json

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tui_gateway.ws import WSTransport, _ws_peer_label, handle_ws

try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:
    _WebSocketDisconnect = Exception


# ---------------------------------------------------------------------------
# WSTransport tests
# ---------------------------------------------------------------------------

@pytest.fixture
def transport_sync():
    """Transport with a fresh event loop for synchronous tests."""
    ws = AsyncMock()
    loop = asyncio.new_event_loop()
    t = WSTransport(ws, loop, peer="test-peer")
    yield t
    loop.close()


@pytest_asyncio.fixture
async def transport():
    """Transport bound to the running pytest-asyncio loop for async tests."""
    ws = AsyncMock()
    loop = asyncio.get_running_loop()
    t = WSTransport(ws, loop, peer="test-peer")
    return t


def test_write_returns_false_when_closed(transport_sync):
    transport_sync._closed = True
    assert transport_sync.write({}) is False


def test_close_sets_closed(transport_sync):
    assert transport_sync._closed is False
    transport_sync.close()
    assert transport_sync._closed is True


@pytest.mark.asyncio
async def test_write_async_returns_false_when_closed(transport):
    transport._closed = True
    result = await transport.write_async({})
    assert result is False


@pytest.mark.asyncio
async def test_safe_send_success(transport):
    transport._ws.send_text = AsyncMock()
    await transport._safe_send("test")
    assert transport._closed is False
    transport._ws.send_text.assert_awaited_once_with("test")


@pytest.mark.asyncio
async def test_safe_send_sets_closed_on_exception(transport):
    transport._ws.send_text = AsyncMock(side_effect=ConnectionError("gone"))
    await transport._safe_send("test")
    assert transport._closed is True


@pytest.mark.asyncio
async def test_write_async_success(transport):
    transport._ws.send_text = AsyncMock()
    result = await transport.write_async({"key": "value"})
    assert result is True
    transport._ws.send_text.assert_awaited_once()


# ---------------------------------------------------------------------------
# _ws_peer_label tests
# ---------------------------------------------------------------------------

def test_peer_label_host_and_port():
    ws = MagicMock()
    ws.client.host = "1.2.3.4"
    ws.client.port = 5678
    assert _ws_peer_label(ws) == "1.2.3.4:5678"


def test_peer_label_host_no_port():
    ws = MagicMock()
    ws.client.host = "1.2.3.4"
    ws.client.port = None
    assert _ws_peer_label(ws) == "1.2.3.4"


def test_peer_label_no_client():
    ws = MagicMock(spec=[])  # no attributes
    assert _ws_peer_label(ws) == "unknown"


# ---------------------------------------------------------------------------
# handle_ws tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_ready_send_fails(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.client.port = 9999
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()

    # Make send_text fail so the ready frame write returns False
    ws.send_text = AsyncMock(side_effect=ConnectionError("refused"))

    await handle_ws(ws)

    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_disconnect(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.client.port = 9999
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()

    # Ready frame succeeds, then receive_text raises disconnect
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock(
        side_effect=_WebSocketDisconnect()
    )

    await handle_ws(ws)

    ws.accept.assert_awaited_once()
    ws.send_text.assert_awaited_once()  # ready frame
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_parse_error(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.client.port = 9999
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()

    # Ready succeeds, then send invalid JSON, then disconnect
    call_count = 0

    async def _receive_text():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "not-valid-json{{"
        raise _WebSocketDisconnect()

    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=_receive_text)

    await handle_ws(ws)

    ws.accept.assert_awaited_once()
    # Two send_text calls: ready frame + parse error response
    assert ws.send_text.await_count == 2
    # Verify parse error response
    second_call_args = ws.send_text.await_args_list[1]
    error_payload = json.loads(second_call_args[0][0])
    assert error_payload["error"]["code"] == -32700


@pytest.mark.asyncio
@patch("tui_gateway.ws.server")
async def test_handle_ws_dispatch_success(mock_server):
    ws = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.client.port = 9999
    mock_server.resolve_skin.return_value = {"theme": "default"}
    mock_server._sessions = {}
    mock_server._stdio_transport = MagicMock()

    rpc_request = {"jsonrpc": "2.0", "method": "test.echo", "id": 1}
    rpc_response = {"jsonrpc": "2.0", "result": "ok", "id": 1}

    call_count = 0

    async def _receive_text():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps(rpc_request)
        raise _WebSocketDisconnect()

    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=_receive_text)
    mock_server.dispatch = MagicMock(return_value=rpc_response)

    await handle_ws(ws)

    ws.accept.assert_awaited_once()
    mock_server.dispatch.assert_called_once()
    # Two send_text calls: ready frame + dispatch response
    assert ws.send_text.await_count == 2
    # Verify response was sent
    second_call_args = ws.send_text.await_args_list[1]
    response_payload = json.loads(second_call_args[0][0])
    assert response_payload["result"] == "ok"
    assert response_payload["id"] == 1
