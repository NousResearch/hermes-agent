"""Tests for tui_gateway.event_publisher -- WS publisher transport.

Covers WsPublisherTransport lifecycle:
- __init__: ws_connect unavailable, connect failure, connect success
- write: dead/ws_none/worker_none guards, successful enqueue, queue full
- close: no worker, graceful drain, ws close with/without errors
- _drain: DRAIN_STOP, non-string, ws none, send success, send failure
"""
from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestWsPublisherTransport:
    """Best-effort WS publisher used by the PTY-side gateway."""

    # ------------------------------------------------------------------ #
    # __init__
    # ------------------------------------------------------------------ #

    def test_init_ws_connect_unavailable(self):
        """When websockets is not installed, the transport starts dead."""
        with patch("tui_gateway.event_publisher.ws_connect", None):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            assert t._dead is True
            assert t._ws is None
            assert t._worker is None

    def test_init_connect_fails(self):
        """When the WS handshake fails, transport starts dead."""
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            side_effect=ConnectionRefusedError("refused"),
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            assert t._dead is True
            assert t._ws is None
            assert t._worker is None

    def test_init_connect_success(self):
        """Successful connect creates a daemon drain thread."""
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            assert t._dead is False
            assert t._ws is mock_ws
            assert t._worker is not None
            assert t._worker.daemon is True
            assert t._worker.name == "hermes-ws-pub"

    # ------------------------------------------------------------------ #
    # write
    # ------------------------------------------------------------------ #

    def test_write_dead(self):
        with patch("tui_gateway.event_publisher.ws_connect", None):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            assert t._dead is True
            assert t.write({"event": "test"}) is False

    def test_write_ws_none(self):
        """When ws is None but dead flag is not set, write returns False."""
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            # Artificially set ws to None
            t._ws = None
            assert t.write({"event": "test"}) is False

    def test_write_worker_none(self):
        """When worker is None but ws exists, write returns False."""
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            # Artificially set worker to None
            t._worker = None
            with patch.object(t._q, "put_nowait") as mock_put:
                result = t.write({"event": "test"})
                # Should return False because worker is None
                assert result is False
                # put_nowait should NOT be called
                mock_put.assert_not_called()

    def test_write_success(self):
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            with patch.object(t._q, "put_nowait") as mock_put:
                result = t.write({"event": "test", "data": 42})
                assert result is True
                mock_put.assert_called_once_with(
                    '{"event": "test", "data": 42}'
                )

    def test_write_queue_full(self):
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            with patch.object(
                t._q, "put_nowait", side_effect=queue.Full
            ):
                result = t.write({"event": "test"})
                assert result is False

    # ------------------------------------------------------------------ #
    # close
    # ------------------------------------------------------------------ #

    def test_close_no_worker(self):
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            t._worker = None
            # Should not raise
            t.close()
            assert t._dead is True
            assert t._worker is None
            # ws should have been closed
            assert mock_ws.close.called

    def test_close_alive_worker_drains(self):
        """close() puts DRAIN_STOP and joins the worker."""
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            t.close()
            assert t._dead is True
            assert t._worker is None

    def test_close_queue_full_on_drain_stop(self):
        """If DRAIN_STOP can't be enqueued, we still proceed."""
        mock_ws = MagicMock()
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            with patch.object(
                t._q, "put_nowait", side_effect=[queue.Full, None]
            ):
                # Should not raise -- first put_nowait (DRAIN_STOP) raises Full
                t.close()
                assert t._dead is True

    def test_close_ws_close_exception(self):
        """If ws.close() raises, close() swallows it."""
        mock_ws = MagicMock()
        mock_ws.close.side_effect = OSError("connection gone")
        with patch(
            "tui_gateway.event_publisher.ws_connect",
            return_value=mock_ws,
        ):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            t.close()
            assert t._dead is True
            assert t._ws is None

    def test_close_ws_already_none(self):
        """close() when ws is already None skips the close call."""
        with patch("tui_gateway.event_publisher.ws_connect", None):
            from tui_gateway.event_publisher import WsPublisherTransport

            t = WsPublisherTransport("ws://localhost:9876/pub")
            t.close()
            assert t._dead is True

    # ------------------------------------------------------------------ #
    # _drain (exercised indirectly via write + close, but test directly)
    # ------------------------------------------------------------------ #

    def test_drain_stop_item_returns(self):
        """_DRAIN_STOP causes _drain to return immediately."""
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        t._ws = MagicMock()
        t._lock = threading.Lock()
        t._dead = False
        t._q.put(_DRAIN_STOP)

        t._drain()
        # Should have returned without calling send
        t._ws.send.assert_not_called()

    def test_drain_non_string_skipped(self):
        """Non-string items from the queue are skipped."""
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        t._ws = MagicMock()
        t._lock = threading.Lock()
        t._dead = False
        t._q.put(42)  # not a string
        t._q.put(_DRAIN_STOP)

        t._drain()
        t._ws.send.assert_not_called()

    def test_drain_ws_none_skips_send(self):
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        t._ws = None
        t._lock = threading.Lock()
        t._dead = False
        t._q.put("hello")
        t._q.put(_DRAIN_STOP)

        t._drain()
        # No send should happen
        assert t._dead is False  # dead should not be set

    def test_drain_send_success(self):
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        t._ws = MagicMock()
        t._lock = threading.Lock()
        t._dead = False
        t._q.put("hello")
        t._q.put(_DRAIN_STOP)

        t._drain()
        t._ws.send.assert_called_once_with("hello")

    def test_drain_send_failure_sets_dead(self):
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        mock_ws = MagicMock()
        mock_ws.send.side_effect = ConnectionError("broken pipe")
        t._ws = mock_ws
        t._lock = threading.Lock()
        t._dead = False
        t._q.put("hello")
        t._q.put(_DRAIN_STOP)

        t._drain()
        assert t._dead is True
        assert t._ws is None

    def test_drain_send_with_lock(self):
        """send is called inside the lock."""
        from tui_gateway.event_publisher import WsPublisherTransport, _DRAIN_STOP

        t = WsPublisherTransport.__new__(WsPublisherTransport)
        t._q = queue.Queue()
        t._ws = MagicMock()
        t._lock = threading.Lock()
        t._dead = False
        t._q.put("test")
        t._q.put(_DRAIN_STOP)

        t._drain()
        t._ws.send.assert_called_once_with("test")
