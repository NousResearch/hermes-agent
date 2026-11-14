"""Idle lease release (#146735) never frees a chat an in-flight writer or turn still needs.

Real SQLite store, real active-session registry, real reaper and real JSON-RPC handlers; the competing
owner is a separate registry lease. Interleavings are pinned with Events at production seams, not sleeps.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tui_gateway import server


class _LiveSocket:
    _closed = False  # the Desktop app is still connected

    def write(self, frame):
        return True


class _Agent:
    def __init__(self, sid: str):
        self.session_id, self.model, self.provider, self.base_url, self.api_key = sid, "old/model", "prov", "", ""
        self.reasoning_config = None
        self.in_switch, self.finish_switch = threading.Event(), threading.Event()

    def clear_interrupt(self):
        return None

    def switch_model(self, **kw):
        self.in_switch.set()
        assert self.finish_switch.wait(10)
        self.model = kw["new_model"]


@pytest.fixture
def chat(monkeypatch):
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    monkeypatch.setattr(server, "_db", db)
    competitors = []
    sid = "idle-race"
    db.create_session(sid, source="desktop")
    ready = threading.Event()
    ready.set()
    long_ago = time.time() - 3600
    record = {
        "session_key": sid, "source": "desktop", "history_version": 0, "history_lock": threading.Lock(),
        "history": [], "agent_ready": ready, "running": False, "created_at": long_ago, "last_active": long_ago,
        "transport": _LiveSocket(), "agent": _Agent(sid), "attached_images": [], "cols": 80}
    server._sessions[sid] = record
    assert server._ensure_active_session_slot(sid, record) is None

    def compete():
        lease, _refusal = try_acquire_active_session(session_id=sid, surface="telegram", config={})
        if lease is not None:
            competitors.append(lease)
        return lease

    try:
        yield SimpleNamespace(sid=sid, record=record, db=db, compete=compete)
    finally:
        record["agent"].finish_switch.set()
        for lease in competitors:
            lease.release()
        server._release_active_session_slot(record)
        server._sessions.pop(sid, None)
        db.close()


def test_reaper_never_releases_a_lease_mid_model_switch(chat, monkeypatch):
    """The reaper's unlocked idle probe ran before the switch refreshed ``last_active``; its release under
    history_lock must re-check the whole predicate and see the switch's claim."""
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="other/model", target_provider="prov"))
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *a, **k: None)
    monkeypatch.setattr(server, "_merge_preflight_warning", lambda *a, **k: None)
    agent = chat.record["agent"]
    responses = []
    switch = threading.Thread(target=lambda: responses.append(server.handle_request({
        "id": "r", "method": "config.set",
        "params": {"session_id": chat.sid, "key": "model", "value": "other/model --provider prov --session"}})))
    pending_kind = server._session_pending_kind
    probes = []

    def probe_then_switch(sid):
        # The last check of the unlocked idle probe: the switch passes its ownership check right here.
        if not probes:
            switch.start()
            assert agent.in_switch.wait(10)
        probes.append(sid)
        return pending_kind(sid)

    monkeypatch.setattr(server, "_session_pending_kind", probe_then_switch)
    server._release_idle_session_leases()
    competitor = chat.compete()
    agent.finish_switch.set()
    switch.join(10)

    assert {"competitor_claimed": competitor is not None, "error": (responses[0] or {}).get("error"),
            "model": agent.model} == {"competitor_claimed": False, "error": None, "model": "other/model"}


@pytest.mark.parametrize("caller", ["session.compress", "slash /compress"])
def test_a_slow_compute_host_compression_keeps_its_lease_until_the_late_ack(chat, monkeypatch, caller):
    late_acks = []

    def host_still_compressing(sid, *, route_name, command="", wait=True, timeout=30.0, on_late_ack=None, **_kw):
        late_acks.append(on_late_ack)
        raise queue.Empty  # the waiter gave up; the host keeps compressing under its borrowed token

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    monkeypatch.setattr(server, "_send_compute_host_control", host_still_compressing)
    if caller == "session.compress":
        result = server.handle_request(
            {"id": "r", "method": "session.compress", "params": {"session_id": chat.sid}})["result"]["status"]
    else:
        result = server._compute_host_slash(chat.sid, chat.record, "compress", "/compress")[0]

    # compression.context_total_ceiling_seconds=900 outlasts the 600s idle window.
    released_mid_compression = server._release_idle_session_leases(time.time() + 900)
    late_acks[0]({"type": "control.error", "message": "host gave up"})
    released_after_ack = server._release_idle_session_leases(time.time() + 900)

    assert (result, released_mid_compression, released_after_ack) == ("pending", 0, 1)


def test_a_refused_loop_tick_never_clears_a_user_turn_that_started_in_the_gap(chat, monkeypatch):
    """Admission clears ``running`` before reporting the refusal; a user prompt can claim the turn right then.
    The tick's rollback restores the loop countdown but must leave that turn's ``running`` alone."""
    from hermes_cli.loops import LoopManager, save_loop

    server._reap_idle_sessions()  # the real idle tick frees the chat
    competitor = chat.compete()
    assert competitor is not None
    manager = LoopManager(chat.sid)
    manager.set("check the build", interval_seconds=1800)
    manager.state.next_due_at = time.time() - 1
    save_loop(chat.sid, manager.state)
    user_turn = []

    def emit(event, sid, payload=None, **_kw):
        if event == "error" and not user_turn:  # the refusal frame: running is already False here
            competitor.release()
            err, _fields = server._lock_in_submit_turn(
                "user", chat.sid, chat.record, "hello", {}, False, None, None, None)
            user_turn.append(err)

    monkeypatch.setattr(server, "_emit", emit)
    try:
        server._maybe_fire_tui_loop_tick(chat.sid, chat.record)
        state = LoopManager(chat.sid).state
        assert {"user_turn_admitted": user_turn == [None], "running": chat.record["running"],
                "awaiting_response": state.awaiting_response} == {
            "user_turn_admitted": True, "running": True, "awaiting_response": False}
    finally:
        chat.record["running"] = False
        manager.clear()
