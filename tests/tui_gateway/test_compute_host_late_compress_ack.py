"""Regression tests for #97948 symptom A (salvaged from #99630).

A manual /compress on a compute-host session used to block its RPC waiter for
a hard-coded 120s, return a 5019 timeout error, and then DROP the host's late
``control.ack`` — so the rotated session_key / history_version / session_info
never reached the gateway session and the desktop never refreshed.
"""

import queue
import sys
import threading
import time
import types

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor


def _supervisor() -> tuple[HostSupervisor, list]:
    sup = HostSupervisor(argv=[sys.executable, "-c", ""], autostart=False)
    sent: list = []
    sup._send_frame = lambda frame: sent.append(frame)
    sup.start = lambda: None  # never spawn a child
    return sup, sent


def _session(**extra) -> dict:
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "old-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 3,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "_compute_host_active": True,
        **extra,
    }


# ── HostSupervisor: late-ack registration ───────────────────────────────────


def test_control_timeout_registers_one_shot_late_ack_handler():
    sup, sent = _supervisor()
    fired: list = []

    with pytest.raises(queue.Empty):
        sup.control("sid", route_name="session.compress", payload={"command": "/compress"},
                    wait=True, timeout=0.05, on_late_ack=fired.append)

    request_id = sent[0]["request_id"]
    assert request_id not in sup._pending_controls
    assert request_id in sup._late_control_handlers

    late = {"type": "control.ack", "request_id": request_id, "result": {"status": "compressed"}}
    sup._handle_host_frame(late)
    assert fired == [late]
    # One-shot: a duplicate ack for the same request is ignored.
    sup._handle_host_frame(late)
    assert fired == [late]
    assert request_id not in sup._late_control_handlers


def test_control_timeout_without_handler_still_drops_late_ack():
    sup, sent = _supervisor()
    with pytest.raises(queue.Empty):
        sup.control("sid", route_name="session.compress", wait=True, timeout=0.05)
    assert sup._late_control_handlers == {}
    sup._handle_host_frame({"type": "control.ack", "request_id": sent[0]["request_id"]})


def test_late_control_error_and_bare_error_frames_fire_handler():
    sup, sent = _supervisor()
    fired: list = []
    for _ in range(2):
        with pytest.raises(queue.Empty):
            sup.control("sid", route_name="session.compress", wait=True, timeout=0.01,
                        on_late_ack=fired.append)
    rid_a, rid_b = sent[0]["request_id"], sent[1]["request_id"]
    sup._handle_host_frame({"type": "control.error", "request_id": rid_a, "message": "boom"})
    sup._handle_host_frame({"type": "error", "request_id": rid_b, "message": "bad frame"})
    assert [f["request_id"] for f in fired] == [rid_a, rid_b]


def test_late_ack_handlers_are_bounded_by_ttl_and_cap(monkeypatch):
    from tui_gateway import host_supervisor as hs

    monkeypatch.setattr(hs, "_LATE_CONTROL_MAX", 3)
    sup, _sent = _supervisor()
    for i in range(5):
        sup._register_late_control_handler(f"r{i}", lambda _f: None)
    assert len(sup._late_control_handlers) == 3
    assert set(sup._late_control_handlers) == {"r2", "r3", "r4"}

    # TTL: an old registration is dropped on the next registration.
    monkeypatch.setattr(hs, "_LATE_CONTROL_TTL_SECS", 0.0)
    time.sleep(0.01)
    sup._register_late_control_handler("fresh", lambda _f: None)
    assert set(sup._late_control_handlers) == {"fresh"}


def test_host_crash_fails_outstanding_late_ack_handlers():
    sup, sent = _supervisor()
    fired: list = []
    with pytest.raises(queue.Empty):
        sup.control("sid", route_name="session.compress", wait=True, timeout=0.01,
                    on_late_ack=fired.append)
    sup._fail_pending_turns(reason="crash", message="compute host exited with code 1")
    assert len(fired) == 1
    assert fired[0]["type"] == "control.error"
    assert fired[0]["request_id"] == sent[0]["request_id"]
    assert sup._late_control_handlers == {}


# ── session.compress RPC: pending answer + late adoption ────────────────────


@pytest.fixture
def compute_host_gateway(monkeypatch):
    sup, sent = _supervisor()
    emitted: list = []
    monkeypatch.setattr(server, "_compute_host_supervisor", sup)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _s, cfg=None: True)
    monkeypatch.setattr(server, "_compute_host_compress_wait_seconds", lambda cfg=None: 0.05)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "mirrored"})
    session = _session()
    server._sessions["sid"] = session
    try:
        yield sup, sent, emitted, session
    finally:
        server._sessions.pop("sid", None)


def _late_ack(request_id: str) -> dict:
    return {
        "type": "control.ack",
        "sid": "sid",
        "request_id": request_id,
        "route_name": "session.compress",
        "result": {"status": "compressed", "removed": 12, "summary": {"headline": "Compressed 14 → 2"}},
        "session_key": "rotated-session-key",
        "history_version": 9,
        "message_count": 2,
        "session_info": {"model": "host-model", "usage": {"total": 111}},
    }


def test_session_compress_reports_pending_and_adopts_late_ack(compute_host_gateway):
    sup, sent, emitted, session = compute_host_gateway

    resp = server.handle_request({"id": "1", "method": "session.compress", "params": {"session_id": "sid"}})

    assert "error" not in resp, resp
    assert resp["result"]["status"] == "pending"
    assert resp["result"]["turn_isolation"] is True
    assert "background" in resp["result"]["message"]
    assert sent[0]["route_name"] == "session.compress"
    # Nothing adopted yet, the host is still working.
    assert session["session_key"] == "old-session-key"
    assert emitted == []

    sup._handle_host_frame(_late_ack(sent[0]["request_id"]))

    assert session["session_key"] == "rotated-session-key"
    assert session["history_version"] == 9
    assert session["_metadata_message_count"] == 2
    assert session["_metadata_mirror"]["model"] == "host-model"
    events = [(event, payload) for event, _sid, payload in emitted]
    assert ("session.info", {"model": "mirrored"}) in events
    assert ("status.update", {"kind": "compacted", "text": "✓ Context compression complete"}) in events


def test_session_compress_late_control_error_surfaces_as_error_event(compute_host_gateway):
    sup, sent, emitted, session = compute_host_gateway

    resp = server.handle_request({"id": "1", "method": "session.compress", "params": {"session_id": "sid"}})
    assert resp["result"]["status"] == "pending"

    sup._handle_host_frame({"type": "control.error", "request_id": sent[0]["request_id"], "message": "provider down"})

    assert session["session_key"] == "old-session-key"
    assert ("error", "sid", {"message": "compression failed: provider down"}) in emitted


def test_session_compress_late_ack_ignored_after_session_closed(compute_host_gateway):
    sup, sent, emitted, session = compute_host_gateway
    server.handle_request({"id": "1", "method": "session.compress", "params": {"session_id": "sid"}})
    server._sessions.pop("sid")

    sup._handle_host_frame(_late_ack(sent[0]["request_id"]))

    assert session["session_key"] == "old-session-key"
    assert emitted == []


def test_slash_compress_route_reports_pending_and_adopts_late_ack(compute_host_gateway):
    sup, sent, emitted, session = compute_host_gateway

    resp = server.handle_request(
        {"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "/compress"}}
    )

    assert "error" not in resp, resp
    assert "compression still running in the background" in resp["result"]["output"]
    assert sent[0]["route_name"] == "slash.compress"

    sup._handle_host_frame({**_late_ack(sent[0]["request_id"]), "route_name": "slash.compress"})
    assert session["session_key"] == "rotated-session-key"
    assert any(event == "session.info" for event, _sid, _p in emitted)


# ── wait budget follows compression.context_total_ceiling_seconds ───────────


def test_compress_wait_budget_follows_config_ceiling():
    assert server._compute_host_compress_wait_seconds({"compression": {}}) == 630.0
    assert server._compute_host_compress_wait_seconds(
        {"compression": {"context_total_ceiling_seconds": 200}}
    ) == 230.0
    # Never below the historical 120s floor, never above the RPC-safe cap.
    assert server._compute_host_compress_wait_seconds(
        {"compression": {"context_total_ceiling_seconds": 10, "context_timeout_seconds": 0}}
    ) == 120.0
    assert server._compute_host_compress_wait_seconds(
        {"compression": {"context_total_ceiling_seconds": 99999}}
    ) == server._COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS
