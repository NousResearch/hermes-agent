"""Tests for concurrent multi-client session attach (channel co-viewing).

Multi-participant channel sessions need several clients watching the SAME
live session at once. The transport layer was built for this — a second
attach grows a :class:`FanoutTransport` instead of stealing the stream, and
``write_json`` resolves the session's transport per-write so a mid-turn
attach starts receiving immediately — but nothing pinned those semantics
until now. These tests are the contract the desktop/cross-device channel
work builds on.
"""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway.transport import FanoutTransport

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


class RecordingTransport:
    """Fake client transport that records every frame it receives."""

    def __init__(self, name: str, alive: bool = True):
        self.name = name
        self.alive = alive
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        if not self.alive:
            return False
        self.frames.append(obj)
        return True

    def close(self) -> None:
        self.alive = False

    def event_types(self) -> list[str]:
        return [
            (frame.get("params") or {}).get("type")
            for frame in self.frames
            if frame.get("method") == "event"
        ]


def _live_session(server, sid: str, transport) -> dict:
    session = {
        "agent": types.SimpleNamespace(model="test/model"),
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "running": False,
        "session_key": f"key-{sid}",
        "transport": transport,
    }
    server._sessions[sid] = session
    return session


def test_second_attach_joins_fanout_instead_of_stealing(server):
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan1", t1)

    server._attach_session_transport(session, t2)

    transport = session["transport"]
    assert isinstance(transport, FanoutTransport)
    assert transport.contains(t1)
    assert transport.contains(t2)


def test_events_broadcast_to_every_attached_client(server):
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan2", t1)
    server._attach_session_transport(session, t2)

    server._emit("message.delta", "chan2", {"text": "hello"})
    server._emit("tool.start", "chan2", {"name": "terminal"})

    assert t1.event_types() == ["message.delta", "tool.start"]
    assert t2.event_types() == ["message.delta", "tool.start"]
    # Same payload object content on both sides.
    assert t1.frames[0]["params"]["payload"] == {"text": "hello"}
    assert t2.frames[0]["params"]["payload"] == {"text": "hello"}


def test_mid_stream_attach_starts_receiving_immediately(server):
    """write_json resolves the session transport PER WRITE, so a client that
    attaches mid-turn sees every subsequent event of that same turn."""
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("late-joiner")
    session = _live_session(server, "chan3", t1)

    server._emit("message.delta", "chan3", {"text": "before join"})
    server._attach_session_transport(session, t2)
    server._emit("message.delta", "chan3", {"text": "after join"})

    assert t1.event_types() == ["message.delta", "message.delta"]
    assert t2.event_types() == ["message.delta"]
    assert t2.frames[0]["params"]["payload"] == {"text": "after join"}


def test_detaching_one_client_keeps_the_session_live_for_others(server):
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan4", t1)
    server._attach_session_transport(session, t2)

    detached = server._detach_transport_from_sessions(t1)

    assert detached == ["chan4"]
    assert server._session_has_live_transport(session)
    server._emit("message.delta", "chan4", {"text": "still flowing"})
    assert t2.event_types() == ["message.delta"]
    assert t1.event_types() == []


def test_detaching_the_last_client_parks_on_the_drop_sentinel(server):
    """Last client gone → the session parks on _detached_ws_transport, NOT real
    stdio: the desktop's in-process gateway has no stdio reader, so stale frames
    would leak into its logs (#38591), and the WS-orphan reaper recognizes the
    sentinel as "detached, safe to grace-reap". A quick reattach still works —
    _attach_session_transport treats the sentinel like an empty slot."""
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan5", t1)
    server._attach_session_transport(session, t2)

    server._detach_transport_from_sessions(t1)
    server._detach_transport_from_sessions(t2)

    assert session["transport"] is server._detached_ws_transport
    assert not server._session_has_live_transport(session)

    # Reattach replaces the sentinel outright (no fanout wrapping).
    t3 = RecordingTransport("desktop-c")
    server._attach_session_transport(session, t3)
    assert session["transport"] is t3
    assert server._session_has_live_transport(session)


def test_dead_client_is_pruned_without_breaking_the_rest(server):
    t1 = RecordingTransport("desktop-a", alive=False)  # peer gone
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan6", t1)
    server._attach_session_transport(session, t2)

    server._emit("message.delta", "chan6", {"text": "hello"})

    assert t2.event_types() == ["message.delta"]
    # The dead transport was pruned from the fanout on write failure.
    assert not session["transport"].contains(t1)
    assert session["transport"].contains(t2)


def test_duplicate_attach_is_idempotent(server):
    t1 = RecordingTransport("desktop-a")
    t2 = RecordingTransport("desktop-b")
    session = _live_session(server, "chan7", t1)
    server._attach_session_transport(session, t2)
    server._attach_session_transport(session, t2)

    server._emit("message.delta", "chan7", {"text": "once"})

    # No double-delivery from a repeated attach of the same client.
    assert t2.event_types() == ["message.delta"]


# -------------------------------------------------------------------------
# prompt.submit sender_device sanitation (channels Phase 2)
# -------------------------------------------------------------------------

def test_sanitize_sender_device(server):
    f = server._sanitize_sender_device
    assert f("omar-iphone") == "omar-iphone"
    assert f("  Omar's   MacBook  Pro  ") == "Omar's MacBook Pro"
    assert f("x" * 500) == "x" * 80
    assert f(None) == ""
    assert f(123) == ""
    assert f(["nope"]) == ""
    assert f("line\nbreaks\tcollapse") == "line breaks collapse"
