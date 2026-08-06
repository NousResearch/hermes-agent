"""Tests for session.observe + write_json observer fan-out.

A live session's event frames normally route to exactly one transport (the
session owner — last resumer wins). ``session.observe`` lets a second-screen
client (LazoChat via the HermesBridge) subscribe to the same frames WITHOUT
stealing ownership; ``write_json`` fans every event out to primary + observers,
and the WS-disconnect teardown sweeps observer attachments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_observe")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._live_transports.clear()
    mod._child_mirrors.clear()
    mod._active_child_runs.clear()


class FakeTransport:
    """Minimal transport recording frames; identity-based like WSTransport."""

    def __init__(self):
        self.frames: list[dict] = []
        self.closed = False

    def write(self, obj: dict) -> bool:
        if self.closed:
            return False
        self.frames.append(obj)
        return True

    def close(self):
        self.closed = True


def _event_frame(sid: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "chunk", "session_id": sid},
    }


def _observe_req(sid: str, observing: bool) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "session.observe",
        "params": {"session_id": sid, "observe": observing},
    }


def test_write_json_fans_to_primary_and_observer(server):
    primary, observer = FakeTransport(), FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": primary}

    assert server.add_session_observer("s1", observer) is True
    server.write_json(_event_frame("s1"))

    assert len(primary.frames) == 1
    assert len(observer.frames) == 1
    assert observer.frames[0]["params"]["session_id"] == "s1"


def test_observer_same_as_primary_not_duplicated(server):
    primary = FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": primary}

    server.add_session_observer("s1", primary)
    server.write_json(_event_frame("s1"))

    assert len(primary.frames) == 1


def test_observe_does_not_steal_ownership(server):
    """The primary transport stays untouched — observers are additive."""
    primary = FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": primary}

    server.add_session_observer("s1", FakeTransport())
    assert server._sessions["s1"]["transport"] is primary


def test_session_observe_rpc_attaches_calling_transport(server):
    primary, observer = FakeTransport(), FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": primary}

    resp = server.dispatch(_observe_req("s1", True), observer)
    assert resp is not None
    assert resp.get("result", {}).get("observing") is True

    server.write_json(_event_frame("s1"))
    assert len(observer.frames) == 1

    resp = server.dispatch(_observe_req("s1", False), observer)
    assert resp is not None
    assert resp.get("result", {}).get("observing") is False

    server.write_json(_event_frame("s1"))
    assert len(observer.frames) == 1  # no new frames after unobserve


def test_observe_accepts_stored_session_key(server):
    observer = FakeTransport()
    server._sessions["abc12345"] = {"session_key": "stored-key", "transport": FakeTransport()}

    resp = server.dispatch(_observe_req("stored-key", True), observer)
    assert resp.get("result", {}).get("session_id") == "abc12345"


def test_observe_unknown_session_errors(server):
    resp = server.dispatch(_observe_req("ghost", True), FakeTransport())
    assert "error" in resp


def test_transport_disconnect_sweeps_observer(server):
    primary, observer = FakeTransport(), FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": primary}
    server.add_session_observer("s1", observer)

    # The single WS-disconnect teardown path (also detaches owned sessions;
    # s1 is observer-only, so only the sweep matters here).
    reaped, detached = server._close_sessions_for_transport(observer, end_reason="ws_disconnect")
    assert (reaped, detached) == (0, 0)

    server.write_json(_event_frame("s1"))
    assert len(primary.frames) == 1
    assert len(observer.frames) == 0


def test_delivery_result_true_when_observer_receives(server):
    """A dead primary must not zero the delivery result if an observer got it."""
    dead = FakeTransport()
    dead.close()
    observer = FakeTransport()
    server._sessions["s1"] = {"session_key": "k1", "transport": dead}
    server.add_session_observer("s1", observer)

    assert server.write_json(_event_frame("s1")) is True
    assert len(observer.frames) == 1


def _locked_session(server, sid: str, transport) -> dict:
    import threading

    session = {"session_key": f"k-{sid}", "transport": transport, "history_lock": threading.Lock()}
    server._sessions[sid] = session
    return session


def test_claim_promotes_displaced_owner_to_observer(server):
    """A client that loses ownership via a re-claim keeps receiving events."""
    old_owner, new_owner = FakeTransport(), FakeTransport()
    session = _locked_session(server, "s1", old_owner)

    with session["history_lock"]:
        server.claim_session_transport(session, new_owner)

    assert server._sessions["s1"]["transport"] is new_owner
    server.write_json(_event_frame("s1"))
    assert len(new_owner.frames) == 1
    assert len(old_owner.frames) == 1  # auto-promoted to observer, still fed


def test_claim_same_transport_is_idempotent(server):
    owner = FakeTransport()
    session = _locked_session(server, "s1", owner)

    with session["history_lock"]:
        server.claim_session_transport(session, owner)

    assert "observer_transports" not in session
    server.write_json(_event_frame("s1"))
    assert len(owner.frames) == 1


def test_claim_none_and_first_claim_do_not_observe(server):
    session = _locked_session(server, "s1", None)

    server.claim_session_transport(session, None)  # no-op
    assert session.get("transport") is None

    owner = FakeTransport()
    with session["history_lock"]:
        server.claim_session_transport(session, owner)

    assert "observer_transports" not in session
    assert session["transport"] is owner


def test_claim_skips_orphan_sentinel(server):
    """The detached-WS sentinel is not a real client — never kept as observer."""
    new_owner = FakeTransport()
    session = _locked_session(server, "s1", server._detached_ws_transport)

    with session["history_lock"]:
        server.claim_session_transport(session, new_owner)

    assert session["transport"] is new_owner
    assert "observer_transports" not in session
