"""``turn.alive``: liveness frames for a running turn that has emitted nothing lately."""

import time

import pytest

from tui_gateway import event_replay, server, turn_alive


class _Transport:
    def __init__(self, delivers=True):
        self.frames = []
        self.delivers = delivers

    def write(self, obj):
        self.frames.append(obj)
        return self.delivers

    def alive(self):
        return [f["params"] for f in self.frames if f.get("params", {}).get("type") == turn_alive.EVENT]


class _Agent:
    _last_activity_desc = "waiting for stream response (60s, prefill)"

    def __init__(self, age_s):
        self._last_activity_ts = time.time() - age_s


@pytest.fixture
def sessions(monkeypatch):
    live: dict[str, dict] = {}
    monkeypatch.setattr(server, "_sessions", live)
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(turn_alive, "_last_emit", {})
    monkeypatch.setattr(turn_alive, "_ended", set())
    event_replay.reset_replay_state()
    yield live
    event_replay.reset_replay_state()


def _session(running=True, **extra):
    return {"transport": _Transport(), "running": running, **extra}


def test_quiet_running_turn_gets_one_frame_per_interval(sessions):
    sessions["s1"] = _session()
    t0 = 1000.0

    assert turn_alive.tick(t0) == 0  # first sight starts the quiet clock
    assert turn_alive.tick(t0 + turn_alive.TURN_ALIVE_INTERVAL_S - 0.1) == 0
    assert turn_alive.tick(t0 + turn_alive.TURN_ALIVE_INTERVAL_S) == 1

    (params,) = sessions["s1"]["transport"].alive()
    assert params["session_id"] == "s1"
    assert params["payload"]["status"] == "working"
    assert params["payload"]["quiet_s"] == turn_alive.TURN_ALIVE_INTERVAL_S


def test_the_frame_itself_restarts_the_quiet_clock(sessions, monkeypatch):
    sessions["s1"] = _session()
    clock = [2000.0]
    monkeypatch.setattr(turn_alive.time, "monotonic", lambda: clock[0])

    turn_alive.tick()
    clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S
    assert turn_alive.tick() == 1
    clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S - 1
    assert turn_alive.tick() == 0
    clock[0] += 1
    assert turn_alive.tick() == 1


def test_any_other_event_for_the_session_defers_it(sessions, monkeypatch):
    sessions["s1"] = _session()
    clock = [3000.0]
    monkeypatch.setattr(turn_alive.time, "monotonic", lambda: clock[0])

    turn_alive.tick()
    clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S - 1
    assert server._emit("message.start", "s1") is True
    clock[0] += 1
    assert turn_alive.tick() == 0
    assert sessions["s1"]["transport"].alive() == []


def test_idle_waiting_detached_and_finalized_sessions_get_nothing(sessions, monkeypatch):
    closed = _Transport()
    closed._closed = True
    sessions["idle"] = _session(running=False)
    sessions["waiting"] = _session()
    # A closed window's turn keeps running on the parked drop sink, which is not None.
    sessions["detached"] = {"transport": server._detached_ws_transport, "running": True}
    sessions["closed"] = {"transport": closed, "running": True}
    sessions["no-transport"] = {"transport": None, "running": True}
    sessions["finalized"] = _session(_finalized=True)
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "approval.request" if sid == "waiting" else "")
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append(sid) or True)

    for n in range(9):  # 40 s of ticks
        turn_alive.tick(n * turn_alive._TICK_S)

    assert emitted == []  # not even attempted: nobody could hear the frame
    assert turn_alive._last_emit == {}


def test_a_frame_the_peer_drops_is_retried_once_per_interval(sessions):
    sessions["s1"] = {"transport": _Transport(delivers=False), "running": True}
    interval = turn_alive.TURN_ALIVE_INTERVAL_S

    turn_alive.tick(0.0)
    assert turn_alive.tick(interval) == 0  # attempted, not delivered
    for n in range(1, 3):  # the next ticks inside the interval do not retry
        assert turn_alive.tick(interval + n * turn_alive._TICK_S) == 0
    turn_alive.tick(2 * interval)

    assert len(sessions["s1"]["transport"].alive()) == 2


def test_a_relayed_compute_host_frame_defers_it(sessions, monkeypatch):
    # An isolated turn's events reach the client through the relay, which writes them without _emit.
    sessions["s1"] = _session(_compute_host_turn_id="t1")
    clock = [4000.0]
    monkeypatch.setattr(turn_alive.time, "monotonic", lambda: clock[0])

    turn_alive.tick()
    clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S - 1
    assert server._relay_compute_host_rpc({"jsonrpc": "2.0", "method": "event", "params": {
        "type": "message.delta", "session_id": "s1", "payload": {"text": "streaming"}}}) is True
    clock[0] += 1

    assert turn_alive.tick() == 0
    assert sessions["s1"]["transport"].alive() == []


def test_an_isolated_turn_reports_the_hosts_activity_not_the_local_agents(sessions):
    sessions["s1"] = _session(agent=_Agent(age_s=900), _compute_host_turn_id="t1",
                              _compute_host_activity_ns=time.perf_counter_ns() - 7_000_000_000)

    turn_alive.tick(0.0)
    turn_alive.tick(turn_alive.TURN_ALIVE_INTERVAL_S)

    payload = sessions["s1"]["transport"].alive()[0]["payload"]
    assert "activity" not in payload  # the label stays in the host process
    assert 6 <= payload["activity_age_s"] <= 9


def test_an_agent_build_counts_as_running(sessions):
    class _NotReady:
        def is_set(self):
            return False

    sessions["s1"] = _session(running=False, agent_ready=_NotReady(), agent_build_started=True)

    turn_alive.tick(0.0)
    assert turn_alive.tick(turn_alive.TURN_ALIVE_INTERVAL_S) == 1
    assert sessions["s1"]["transport"].alive()[0]["payload"]["status"] == "starting"


def test_carries_the_local_agents_activity(sessions):
    sessions["s1"] = _session(agent=_Agent(age_s=42))

    turn_alive.tick(0.0)
    turn_alive.tick(turn_alive.TURN_ALIVE_INTERVAL_S)

    payload = sessions["s1"]["transport"].alive()[0]["payload"]
    assert payload["activity"] == "waiting for stream response (60s, prefill)"
    assert 41 <= payload["activity_age_s"] <= 44


def test_frames_are_not_sequenced_or_kept_for_replay(sessions, monkeypatch):
    sessions["s1"] = _session()
    clock = [5000.0]
    monkeypatch.setattr(turn_alive.time, "monotonic", lambda: clock[0])
    server._emit("message.start", "s1")

    for _ in range(39):  # a ten-minute quiet tool call
        clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S
        turn_alive.tick()

    alive = sessions["s1"]["transport"].alive()
    assert len(alive) == 39
    assert all("seq" not in params for params in alive)
    assert event_replay.latest_seq("s1") == 1
    assert [e["type"] for e in event_replay.events_since("s1", 0)] == ["message.start"]


def test_a_later_turn_counts_its_quiet_afresh(sessions):
    sessions["s1"] = _session()
    turn_alive.tick(0.0)

    sessions["s1"]["running"] = False
    turn_alive.tick(100.0)
    sessions["s1"]["running"] = True

    assert turn_alive.tick(200.0) == 0  # first sight of the new turn
    assert turn_alive.tick(200.0 + turn_alive.TURN_ALIVE_INTERVAL_S) == 1


def test_no_frame_once_the_turn_sent_message_complete(sessions, monkeypatch):
    """``running`` outlives ``message.complete`` (goal judge, loop hooks, follow-up scheduling). A frame in that
    window reached a Desktop that had already settled the turn, so it read as the finished turn still going."""
    sessions["s1"] = _session()
    clock = [6000.0]
    monkeypatch.setattr(turn_alive.time, "monotonic", lambda: clock[0])
    server._emit("message.start", "s1")
    turn_alive.tick()
    server._emit("message.complete", "s1", {"text": "done", "status": "complete"})

    for _ in range(12):  # a minute of post-turn work with running still True
        clock[0] += turn_alive._TICK_S
        assert turn_alive.tick() == 0
    assert sessions["s1"]["transport"].alive() == []

    # The next turn's message.start is a new turn: its quiet counts afresh and gets frames again.
    server._emit("message.start", "s1")
    clock[0] += turn_alive.TURN_ALIVE_INTERVAL_S
    assert turn_alive.tick() == 1


def test_a_turn_finishing_mid_tick_gets_no_late_frame(sessions, monkeypatch):
    """The ticker found the session due, then the turn thread sent message.complete and cleared running before
    the frame went out: nothing may follow the terminal frame on the wire."""
    sessions["s1"] = _session()
    turn_alive.tick(0.0)
    activity = turn_alive._activity

    def finish_while_building_the_frame(session):
        server._emit("message.complete", "s1", {"text": "done", "status": "complete"})
        session["running"] = False
        return activity(session)

    monkeypatch.setattr(turn_alive, "_activity", finish_while_building_the_frame)

    assert turn_alive.tick(turn_alive.TURN_ALIVE_INTERVAL_S) == 0
    types = [f["params"]["type"] for f in sessions["s1"]["transport"].frames]
    assert types == ["message.complete"]


def test_payload_matches_the_contract(sessions, monkeypatch):
    from tui_gateway.contracts import registry

    monkeypatch.setattr(registry, "STRICT", True)
    sessions["s1"] = _session(agent=_Agent(age_s=5))

    turn_alive.tick(0.0)
    assert turn_alive.tick(turn_alive.TURN_ALIVE_INTERVAL_S) == 1


def test_interval_stays_inside_the_clients_silence_window():
    """Desktop builds hardcode a 45 s silence window and never read ``turn_alive_s``: the frame after a lost
    one (2 x interval + one tick) must still land inside it."""
    assert turn_alive.TURN_ALIVE_INTERVAL_S == 15.0
    assert 2 * turn_alive.TURN_ALIVE_INTERVAL_S + turn_alive._TICK_S < 45.0


def test_interval_is_not_an_env_setting():
    """Behavioural knobs belong in ``config.yaml`` (AGENTS.md), and this one is not a knob at all: a stray
    ``HERMES_TURN_ALIVE_S`` in the environment must neither stretch nor disable the frames."""
    import os
    import subprocess
    import sys

    for value in ("0", "60"):
        out = subprocess.run(
            [sys.executable, "-c", "from tui_gateway import turn_alive; print(turn_alive.TURN_ALIVE_INTERVAL_S)"],
            capture_output=True, text=True, check=True, env={**os.environ, "HERMES_TURN_ALIVE_S": value},
        )
        assert out.stdout.strip() == "15.0"
