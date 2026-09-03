"""Tests for tui_gateway/ws.py — WebSocket transport layer.

Covers WSTransport, _ws_peer_label, _disable_nagle, and handle_ws."""

from __future__ import annotations

import asyncio
import json
import socket
from concurrent.futures import Future, TimeoutError
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from tui_gateway.ws import WSTransport, _disable_nagle, _ws_peer_label


# Sentinel for WebSocketDisconnect — matches what ws.py does on import failure
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:
    _WebSocketDisconnect = Exception


class FakeWebSocket:
    """Minimal stand-in for a starlette WebSocket to avoid starlette imports in test."""

    def __init__(self, host="127.0.0.1", port=12345):
        self.client = FakeClient(host, port)
        self.scope = {"extensions": {"transport": FakeTransport()}}
        self._transport = FakeTransport()
        self.send_text = AsyncMock()
        self.receive_text = AsyncMock()
        self.accept = AsyncMock()

    async def accept(self):
        pass


class FakeClient:
    def __init__(self, host: str, port: int | None):
        self.host = host
        self.port = port


class FakeTransport:
    def __init__(self):
        self._sock = MagicMock(spec=socket.socket)

    def get_extra_info(self, key: str):
        if key == "socket":
            return self._sock
        return None


class TestWSTransport:
    """Tests for the WSTransport class in isolation."""

    @pytest.fixture
    def ws(self):
        return FakeWebSocket()

    @pytest.fixture
    def transport(self, ws):
        loop = asyncio.new_event_loop()
        return WSTransport(ws, loop, peer="127.0.0.1:12345")

    # --- __init__ ---

    def test_init_sets_attrs(self):
        ws = FakeWebSocket()
        loop = asyncio.new_event_loop()
        t = WSTransport(ws, loop, peer="1.2.3.4:5678")
        assert t._ws is ws
        assert t._loop is loop
        assert t._peer == "1.2.3.4:5678"
        assert t._closed is False

    def test_init_defaults_peer_to_unknown(self):
        ws = FakeWebSocket()
        loop = asyncio.new_event_loop()
        t = WSTransport(ws, loop)
        assert t._peer == "unknown"

    # --- write (off-loop thread) ---

    def test_write_returns_false_when_closed(self, transport):
        transport._closed = True
        assert transport.write({"test": "data"}) is False

    def test_write_off_loop_invokes_safe_schedule(self, transport, ws):
        """write() from a non-loop thread uses safe_schedule_threadsafe."""
        fut = Future()
        fut.set_result(None)

        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=fut) as mock_sst:
            result = transport.write({"msg": "hello"})

        assert result is True
        mock_sst.assert_called_once()
        assert transport._closed is False

    def test_write_safe_schedule_returns_none_closes(self, transport):
        """If safe_schedule_threadsafe returns None, the transport is marked closed."""
        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=None):
            result = transport.write({"msg": "hello"})

        assert result is False
        assert transport._closed is True

    def test_write_catches_timeout_keeps_alive(self, transport):
        """TimeoutError should not close the transport."""
        def _timeout(*args, **kwargs):
            raise TimeoutError("stalled")

        with patch("agent.async_utils.safe_schedule_threadsafe") as mock_sst:
            mock_fut = MagicMock()
            mock_fut.result = _timeout
            mock_sst.return_value = mock_fut

            result = transport.write({"data": "x"})
            assert result is True
            assert transport._closed is False

    def test_write_returns_false_on_unknown_exception(self, transport):
        """Unexpected exception closes the transport."""
        with patch("agent.async_utils.safe_schedule_threadsafe") as mock_sst:
            mock_fut = MagicMock()
            mock_fut.result = MagicMock(side_effect=RuntimeError("socket gone"))
            mock_sst.return_value = mock_fut

            result = transport.write({"fail": True})
            assert result is False
            assert transport._closed is True

    # --- write_async ---

    @pytest.mark.asyncio
    async def test_write_async_returns_false_when_closed(self, transport):
        transport._closed = True
        result = await transport.write_async({"key": "val"})
        assert result is False

    @pytest.mark.asyncio
    async def test_write_async_sends_and_returns_true(self, transport, ws):
        result = await transport.write_async({"ok": True})
        assert result is True
        expected = json.dumps({"ok": True}, ensure_ascii=False)
        ws.send_text.assert_awaited_once_with(expected)

    @pytest.mark.asyncio
    async def test_write_async_send_failure_closes_transport(self, transport, ws):
        ws.send_text.side_effect = RuntimeError("socket gone")
        result = await transport.write_async({"fail": True})
        assert result is False
        assert transport._closed is True

    # --- _safe_send ---

    @pytest.mark.asyncio
    async def test_safe_send_success(self, transport, ws):
        await transport._safe_send("{}")
        ws.send_text.assert_awaited_once_with("{}")
        assert transport._closed is False

    @pytest.mark.asyncio
    async def test_safe_send_failure_closes_transport(self, transport, ws):
        ws.send_text.side_effect = ConnectionError("reset")
        await transport._safe_send("{}")
        assert transport._closed is True

    # --- close ---

    def test_close_sets_closed_flag(self, transport):
        assert transport._closed is False
        transport.close()
        assert transport._closed is True


class TestWSPeerLabel:
    """Tests for _ws_peer_label."""

    def test_with_client_attrs(self):
        ws = MagicMock()
        ws.client.host = "10.0.0.1"
        ws.client.port = 8080
        assert _ws_peer_label(ws) == "10.0.0.1:8080"

    def test_without_port(self):
        ws = MagicMock()
        ws.client.host = "10.0.0.1"
        ws.client.port = None
        assert _ws_peer_label(ws) == "10.0.0.1"

    def test_unknown_when_no_client(self):
        ws = MagicMock(spec=["transport"])
        ws.client = None
        assert _ws_peer_label(ws) == "unknown"

    def test_unknown_when_no_client_attrs(self):
        ws = MagicMock()
        ws.client.host = None
        ws.client.port = None
        assert _ws_peer_label(ws) == "unknown"


class TestDisableNagle:
    """Tests for _disable_nagle."""

    def test_sets_tcp_nodelay(self):
        ws = FakeWebSocket()
        _disable_nagle(ws)
        # The transport is resolved from scope["extensions"]["transport"]
        ext_transport = ws.scope["extensions"]["transport"]
        ext_transport._sock.setsockopt.assert_called_once_with(
            socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
        )

    def test_skips_if_no_socket(self):
        ws = MagicMock()
        ws.scope = {}
        type(ws).transport = PropertyMock(return_value=None)
        _disable_nagle(ws)

    def test_skips_on_exception(self):
        ws = MagicMock()
        ws.scope = {"extensions": "not-a-dict"}
        _disable_nagle(ws)


class TestHandleWs:
    """Tests for handle_ws — the main WebSocket session handler."""

    @pytest.mark.asyncio
    async def test_accepts_and_sends_ready(self):
        ws = FakeWebSocket(host="test", port=None)

        with patch("tui_gateway.ws.server") as mock_server:
            mock_server.dispatch = MagicMock(return_value=None)
            mock_server.resolve_skin.return_value = "test-skin"
            mock_server._close_sessions_for_transport = MagicMock(return_value=(0, 0))

            from tui_gateway.ws import handle_ws

            ws.receive_text.side_effect = _WebSocketDisconnect()
            await handle_ws(ws)

        ws.accept.assert_awaited_once()
        assert ws.send_text.await_count >= 1
        ready_call = ws.send_text.await_args_list[0][0][0]
        ready = json.loads(ready_call)
        assert ready["method"] == "event"
        assert ready["params"]["type"] == "gateway.ready"

    @pytest.mark.asyncio
    async def test_sends_parse_error_for_invalid_json(self):
        ws = FakeWebSocket(host="test", port=None)
        ws.receive_text.side_effect = ["not-json", _WebSocketDisconnect()]

        with patch("tui_gateway.ws.server") as mock_server:
            mock_server.dispatch = MagicMock(return_value=None)
            mock_server.resolve_skin.return_value = "skin"
            mock_server._close_sessions_for_transport = MagicMock(return_value=(0, 0))

            from tui_gateway.ws import handle_ws
            await handle_ws(ws)

        assert ws.send_text.await_count >= 2
        error_sent = False
        for call in ws.send_text.await_args_list:
            text = call[0][0]
            if "-32700" in text:
                error_sent = True
                break
        assert error_sent, "Expected parse error (-32700) response"

    @pytest.mark.asyncio
    async def test_dispatches_valid_request(self):
        ws = FakeWebSocket(host="test", port=None)

        valid_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        ws.receive_text.side_effect = [valid_req, _WebSocketDisconnect()]

        with patch("tui_gateway.ws.server") as mock_server:
            mock_server.resolve_skin.return_value = "skin"
            mock_server._close_sessions_for_transport = MagicMock(return_value=(0, 0))
            mock_server.dispatch = MagicMock(return_value=None)

            from tui_gateway.ws import handle_ws
            await handle_ws(ws)

        mock_server.dispatch.assert_called_once()
        req_arg = mock_server.dispatch.call_args[0][0]
        assert req_arg["method"] == "ping"
        assert req_arg["id"] == 1

    @pytest.mark.asyncio
    async def test_ready_send_failure_returns_early(self):
        ws = FakeWebSocket(host="test", port=None)
        ws.send_text.side_effect = RuntimeError("send failed")

        with patch("tui_gateway.ws.server") as mock_server:
            mock_server.resolve_skin.return_value = "skin"
            mock_server._close_sessions_for_transport = MagicMock(return_value=(0, 0))
            mock_server.dispatch = MagicMock(return_value=None)

            from tui_gateway.ws import handle_ws
            await handle_ws(ws)

        ws.accept.assert_awaited_once()
        assert ws.send_text.await_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_crash_sends_internal_error(self):
        ws = FakeWebSocket(host="test", port=None)

        valid_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "crash"})
        ws.receive_text.side_effect = [valid_req, _WebSocketDisconnect()]

        with patch("tui_gateway.ws.server") as mock_server:
            mock_server.resolve_skin.return_value = "skin"
            mock_server._close_sessions_for_transport = MagicMock(return_value=(0, 0))
            mock_server.dispatch = MagicMock(side_effect=ValueError("boom"))

            from tui_gateway.ws import handle_ws
            await handle_ws(ws)

        error_sent = False
        for call in ws.send_text.await_args_list:
            text = call[0][0]
            if "-32603" in text:
                error_sent = True
                break
        assert error_sent, "Expected internal error (-32603) response"
