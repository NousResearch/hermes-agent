"""Regression tests for #97948 symptom A (salvaged from #99630).

A manual /compress on a compute-host session used to block its RPC waiter for
a hard-coded 120s, return a 5019 timeout error, and then DROP the host's late
``control.ack`` — so the rotated session_key / history_version / session_info
never reached the gateway session and the desktop never refreshed.
"""

import json
import queue
import sys
import threading
import time
import types

import pytest

from tests.waiting import wait_until
from tui_gateway import host_supervisor, server
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


class _ExitedHost:
    def wait(self):
        return 1


def _crash(sup: HostSupervisor) -> None:
    """Run the supervisor's own exit path for a host that exited with code 1."""
    sup._proc = _ExitedHost()
    sup._remove_registry = lambda: None
    sup._maybe_respawn_after_crash = lambda: None
    sup._wait_for_exit(sup._proc)


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
    _crash(sup)
    assert len(fired) == 1
    assert fired[0]["type"] == "control.error"
    assert fired[0]["request_id"] == sent[0]["request_id"]
    assert sup._late_control_handlers == {}


def test_host_crash_answers_controls_still_waiting():
    """A control or respond call blocked on the host's reply returns an error as soon as the host
    exits, instead of waiting out its whole timeout for a child that is gone."""
    sup, sent = _supervisor()
    replies: list = []
    callers = [
        threading.Thread(target=lambda: replies.append(sup.control(
            "sid", route_name="session.compress", wait=True, timeout=30))),
        threading.Thread(target=lambda: replies.append(sup.respond("sid", {"cancel": {}}, timeout=30))),
        threading.Thread(target=lambda: replies.append(sup.reload_mcp("sid"))),
    ]
    for caller in callers:
        caller.start()
    assert wait_until(lambda: len(sup._pending_controls) == 3, timeout=5)

    _crash(sup)
    for caller in callers:
        caller.join(timeout=5)

    assert sorted(r["type"] for r in replies) == ["control.error", "control.error", "respond.error"]
    assert {r["request_id"] for r in replies} == {f["request_id"] for f in sent}
    assert all(r["message"] == "compute host exited with code 1" for r in replies)


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


def test_compress_wait_budget_follows_config_ceiling(monkeypatch):
    import agent.auxiliary_client as aux

    # The aux compression request budget floors the idle window and, with it, the ceiling (#114594): the
    # RPC waiter must outlast a compaction the host is still allowed to run.
    monkeypatch.setattr(aux, "_effective_aux_timeout", lambda task, timeout: 300.0)
    assert server._compute_host_compress_wait_seconds(
        {"compression": {"context_total_ceiling_seconds": 200}}
    ) == 330.0
    # Legacy clamps, judged with the aux floor pinned off.
    monkeypatch.setattr(aux, "_effective_aux_timeout", lambda task, timeout: 0.0)
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


def test_reload_mcp_reports_a_host_crash_as_an_error(compute_host_gateway):
    """A crash answers the waiting reload with control.error; the RPC must not report "reloaded"."""
    sup, _sent, _emitted, _session = compute_host_gateway
    replies: list = []
    caller = threading.Thread(target=lambda: replies.append(server.handle_request(
        {"id": "9", "method": "reload.mcp", "params": {"session_id": "sid", "confirm": True}})))
    caller.start()
    assert wait_until(lambda: sup._pending_controls, timeout=5)

    _crash(sup)
    caller.join(timeout=5)

    assert replies[0]["error"]["code"] == 5019
    assert "compute host exited with code 1" in replies[0]["error"]["message"]


def test_a_crash_does_not_fail_controls_sent_to_the_replacement_host():
    """Waiters are detached with the exited process; a control registered afterwards, while a turn
    callback has already started a replacement host, keeps waiting for that host's reply."""
    sup, sent = _supervisor()
    replies: list = []

    def start_a_new_control(_frame):
        threading.Thread(target=lambda: replies.append(sup.control(
            "sid", route_name="session.save", wait=True, timeout=5))).start()
        assert wait_until(lambda: any(f.get("route_name") == "session.save" for f in sent), timeout=5)

    sup._pending_turns["turn-1"] = ("sid", start_a_new_control)
    _crash(sup)

    new_request = sent[-1]["request_id"]
    assert replies == []
    sup._handle_host_frame({"type": "control.ack", "sid": "sid", "request_id": new_request, "route_name": "session.save"})
    assert wait_until(lambda: replies, timeout=5)[0]["type"] == "control.ack"


def test_reload_mcp_reports_an_error_inside_the_host_ack(compute_host_gateway):
    """The host wraps its own reload.mcp reply in reload_mcp.ack even when that reply is an error."""
    sup, sent, _emitted, _session = compute_host_gateway
    replies: list = []
    caller = threading.Thread(target=lambda: replies.append(server.handle_request(
        {"id": "9", "method": "reload.mcp", "params": {"session_id": "sid", "confirm": True}})))
    caller.start()
    assert wait_until(lambda: sent, timeout=5)
    sup._handle_host_frame({"type": "reload_mcp.ack", "sid": "sid", "request_id": sent[-1]["request_id"],
                            "response": {"id": "9", "error": {"code": 5000, "message": "discovery failed"}}})
    caller.join(timeout=5)

    assert replies[0]["error"]["code"] == 5019
    assert "discovery failed" in replies[0]["error"]["message"]


def test_a_crash_answers_waiting_controls_before_the_turn_callbacks_run():
    """A turn callback can run long; a control waiting on the crashed host is already answered by
    then, so it never times out into a late handler that the crash cleanup has passed."""
    sup, _sent = _supervisor()
    replies: list = []
    seen_by_callback: list = []
    caller = threading.Thread(target=lambda: replies.append(sup.control(
        "sid", route_name="session.compress", wait=True, timeout=30)))
    caller.start()
    assert wait_until(lambda: sup._pending_controls, timeout=5)

    def turn_callback(_frame):
        caller.join(timeout=5)
        seen_by_callback.append(list(replies))

    sup._pending_turns["turn-1"] = ("sid", turn_callback)
    _crash(sup)

    assert seen_by_callback == [[{"type": "control.error", "request_id": seen_by_callback[0][0]["request_id"],
                                  "reason": "crash", "message": "compute host exited with code 1"}]]


def test_an_ack_the_host_wrote_before_exiting_still_answers_its_waiter(monkeypatch):
    """The host can write a control.ack and exit right away (a SIGTERM drain, a crash after the
    reply), and the stdout drain may read that ack after the exit is seen: its waiter gets the ack,
    not the crash error. A control sent to a replacement host meanwhile keeps waiting."""
    sup, _sent = _supervisor()
    exit_now, cleaned_up = threading.Event(), threading.Event()
    replies: dict = {}

    def send(request_id):
        threading.Thread(target=lambda: replies.setdefault(request_id, sup.control(
            "sid", route_name="session.save", payload={"request_id": request_id}, wait=True, timeout=30))).start()
        assert wait_until(lambda: request_id in sup._pending_controls, timeout=5)

    def host_stdout():
        yield json.dumps({"type": "hello", "build_sha": "", "hermes_home": ""}) + "\n"
        assert wait_until(lambda: sup._proc is None, timeout=5)  # the exit is already seen
        send("to-replacement")
        yield json.dumps({"type": "control.ack", "sid": "sid", "request_id": "save-1", "result": {}}) + "\n"

    class _HostThatAckedThenExited:
        pid, stdin, stdout, stderr = 0, None, host_stdout(), iter(())

        def wait(self):
            exit_now.wait(timeout=5)
            return 0

    monkeypatch.setattr(host_supervisor.subprocess, "Popen", lambda *_a, **_k: _HostThatAckedThenExited())
    sup._persist_registry = sup._remove_registry = lambda: None
    sup._maybe_respawn_after_crash = cleaned_up.set
    sup._spawn_locked(reason="test")
    send("save-1")

    exit_now.set()
    assert cleaned_up.wait(timeout=10)

    assert wait_until(lambda: "save-1" in replies, timeout=5)
    assert replies["save-1"]["type"] == "control.ack"
    assert "to-replacement" not in replies and "to-replacement" in sup._pending_controls
