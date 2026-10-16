"""A backend-reclaimed session must tell the clients still holding it.

The idle-TTL reaper, the LRU cap, and the WS-orphan reap all tear a live
session down without the client asking. Before ``session.reclaimed`` the client
kept a runtime id the backend had already forgotten, and only discovered it by
failing a later prompt — the "sessions suddenly lost in the backend" report.

Contracts here: a reclaim broadcasts with its reason, a user-initiated close
stays silent, and a notify failure never breaks teardown.
"""

import pytest

from tui_gateway import server


@pytest.fixture()
def captured(monkeypatch):
    events = []
    monkeypatch.setattr(
        server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload))
    )
    # Teardown's real work (finalize, agent close, notifier unregister) is out
    # of scope — this is about what reaches the client.
    monkeypatch.setattr(server, "_finalize_session", lambda *a, **k: None)
    return events


def _session():
    return {"_sid": "live-abc", "session_key": "20260731_120000_aaaaaa"}


@pytest.mark.parametrize("reason", ["idle_timeout", "lru_evict", "ws_orphan_reap"])
def test_reclaim_reasons_announce_to_clients(captured, reason):
    server._teardown_session(_session(), end_reason=reason)

    assert captured == [
        (
            "session.reclaimed",
            {
                "session_id": "live-abc",
                "stored_session_id": "20260731_120000_aaaaaa",
                "reason": reason,
            },
        )
    ]


@pytest.mark.parametrize("reason", ["tui_close", "tui_shutdown", "ws_disconnect", "branched"])
def test_client_initiated_closes_stay_silent(captured, reason):
    """The client asked for these, so announcing them would be noise."""
    server._teardown_session(_session(), end_reason=reason)

    assert captured == []


def test_broadcast_failure_does_not_break_teardown(monkeypatch):
    """A wedged peer must not leave a session half torn down."""
    finalized = []

    def _boom(*_a, **_k):
        raise RuntimeError("transport gone")

    monkeypatch.setattr(server, "_broadcast_global_event", _boom)
    monkeypatch.setattr(
        server, "_finalize_session", lambda s, **k: finalized.append(k.get("end_reason"))
    )

    server._teardown_session(_session(), end_reason="ws_orphan_reap")

    assert finalized == ["ws_orphan_reap"]


def test_reap_paths_stamp_the_runtime_id_the_client_holds():
    """_pop_session_by_id stamps ``_sid``; the payload is useless without it.

    All three reclaim paths pop before tearing down, so this is the invariant
    that makes the broadcast addressable on the client.
    """
    server._sessions["live-xyz"] = {"session_key": "k"}
    try:
        popped = server._pop_session_by_id("live-xyz")
    finally:
        server._sessions.pop("live-xyz", None)

    assert popped is not None
    assert popped["_sid"] == "live-xyz"
