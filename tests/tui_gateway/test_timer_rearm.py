"""/heartbeat and /loop re-arming for TUI/Desktop sessions (#103044, #102056, #102088).

The process that persists the timer is not always the process that runs the turn: Desktop's slash
worker queues the wakeup into a ``_pending_input`` no loop drains, and a gateway/slash worker that
dies mid-turn leaves the tick claimed. The session-owner poller
(``tui_gateway.session_notifications._notification_poller_loop``) is the existing watcher for these
sessions, so it has to re-fire a due timer and re-arm the next period: a timer that fires once
(``ticks_fired=1``, ``awaiting_response`` still set, ``next_due_at`` stale in the past) and then goes
permanently silent is the bug. Due-ness is asserted on an injected clock (``is_due(now)`` /
``is_due(now=...)``) so the timing is exact rather than wall-clock raced.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CADENCE = 60
OVERDUE = 4219  # the "due_in = -4219s" evidence in the report


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()


@pytest.fixture()
def session(server):
    sid, key = "sid-timer-test", "tui-timer-session-1"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock()}
    server._sessions[sid] = s
    server._get_db().create_session(key, source="desktop")
    return sid, key, s


def _store_loop(key: str, *, ticks_fired: int = 1, awaiting: bool, fired_ago: float, due_ago: float) -> float:
    """Persist a loop row; offsets are seconds in the past. Returns the clock it was built on."""
    from hermes_cli.loops import LoopManager, save_loop

    now = time.time()
    st = LoopManager(key).set("check the backend", interval_seconds=CADENCE)
    st.created_at = now - fired_ago - CADENCE
    st.last_fired_at = now - fired_ago
    st.next_due_at = now - due_ago
    st.ticks_fired, st.awaiting_response = ticks_fired, awaiting
    save_loop(key, st)
    return now


def _fire(server, sid, s, times: int = 1):
    """Direct passes of the two timer drivers the poller loop calls; returns the injected prompts."""
    prompts: list[str] = []

    def submit(rid, sid_, session_, text, **kw):
        prompts.append(text)
        return True

    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        for _ in range(times):
            server._maybe_fire_tui_loop_tick(sid, s)
            server._maybe_fire_tui_heartbeat_tick(sid, s)
    return prompts


def test_poller_refires_and_rearms_a_loop_whose_wakeup_turn_never_completed(server, session):
    """The reported payload (ticks_fired=1, awaiting_response=True, next_due_at 70 min stale) is
    driven by the existing per-session poller and re-armed — not silent forever."""
    from hermes_cli.loops import load_loop

    sid, key, s = session
    _store_loop(key, awaiting=True, fired_ago=OVERDUE, due_ago=OVERDUE - CADENCE)
    prompts: list[str] = []

    def submit(rid, sid_, session_, text, **kw):
        prompts.append(text)
        return True

    stop = threading.Event()
    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        t = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, s), daemon=True)
        t.start()
        deadline = time.monotonic() + 8
        while not prompts and time.monotonic() < deadline:
            time.sleep(0.1)
        stop.set()
        t.join(timeout=5)

    assert len(prompts) == 1 and "check the backend" in prompts[0]
    st = load_loop(key)
    assert st.ticks_fired == 2  # the missed period fired exactly once
    assert st.next_due_at > time.time()  # and the next period is armed, not stale
    assert st.awaiting_response is True and s["running"] is True  # claimed for the wakeup turn


def test_stale_wakeup_claim_is_stale_on_the_injected_clock(server, session):
    """The saved state from the report must read due once its claim has outlived the cadence; an
    in-flight claim must not."""
    from hermes_cli.loops import LoopManager, load_loop, save_loop

    _, key, _s = session
    now = _store_loop(key, awaiting=True, fired_ago=OVERDUE, due_ago=OVERDUE - CADENCE)
    mgr = LoopManager(key)

    assert mgr.is_due(now) is True  # the -4219s row: due, not parked on a lost claim
    wakeup = mgr.fire_tick()
    assert wakeup is not None and "check the backend" in wakeup
    st = load_loop(key)
    assert st.ticks_fired == 2 and st.awaiting_response is True
    assert st.next_due_at >= time.time() + CADENCE - 5  # re-armed to the next period
    assert LoopManager(key).is_due(time.time()) is False

    # A claim from a turn that really is still running keeps the tick shut (no double fire).
    live = now
    st = load_loop(key)
    st.last_fired_at, st.next_due_at, st.awaiting_response = live, live - 1, True
    save_loop(key, st)
    assert LoopManager(key).is_due(live) is False
    assert LoopManager(key).fire_tick() is None


def test_overdue_loop_catches_up_once_and_does_not_replay_missed_periods(server, session):
    """A long overdue loop fires ONE catch-up tick, and the turn's end re-arms from now — the ~70
    missed periods are coalesced, not replayed as a burst."""
    from hermes_cli.loops import LoopManager, load_loop

    _, key, _s = session
    now = _store_loop(key, ticks_fired=1, awaiting=False, fired_ago=OVERDUE, due_ago=OVERDUE - CADENCE)
    mgr = LoopManager(key)

    assert mgr.is_due(now) is True
    assert mgr.fire_tick() is not None
    assert load_loop(key).ticks_fired == 2  # one catch-up, not ~70

    decision = LoopManager(key).complete_tick("still checking")
    assert decision["status"] == "active" and decision["stopped"] is False
    st = load_loop(key)
    assert st.awaiting_response is False and st.ticks_fired == 2
    assert abs(st.next_due_at - (time.time() + CADENCE)) <= 5  # next period armed from turn end
    assert LoopManager(key).is_due(time.time()) is False


def test_state_is_due_matches_the_manager_gate(server, session):
    """One rule for every driver: the module-level predicate (the gateway scans rows with it) and
    ``LoopManager.is_due`` agree on the abandoned-claim case."""
    from hermes_cli.loops import LoopManager, state_is_due

    _, key, _s = session
    now = _store_loop(key, awaiting=True, fired_ago=OVERDUE, due_ago=OVERDUE - CADENCE)
    state = LoopManager(key).state
    assert state_is_due(state, now) is True
    assert state_is_due(state, now) == LoopManager(key).is_due(now)
    state.status = "paused"
    assert state_is_due(state, now) is False


def test_session_without_timer_state_polls_and_touches_nothing(server, session):
    """Protection: a session with no /loop and no /heartbeat fires nothing and writes nothing."""
    from hermes_cli.heartbeat import load_heartbeat
    from hermes_cli.loops import load_loop

    sid, key, s = session
    assert _fire(server, sid, s, times=3) == []
    assert s["running"] is False
    assert load_loop(key) is None and load_heartbeat(key) is None


def test_overdue_heartbeat_fires_once_and_reanchors(server, session):
    """/heartbeat after a long outage injects one prompt and re-anchors on the injected clock."""
    from hermes_cli.heartbeat import HeartbeatManager, load_heartbeat, save_heartbeat

    sid, key, s = session
    now = time.time() - OVERDUE
    mgr = HeartbeatManager(key)
    st = mgr.set("report backend health", CADENCE)
    st.created_at, st.last_fired_at = now - 2 * CADENCE, now - CADENCE
    save_heartbeat(key, st)

    loaded = load_heartbeat(key)
    assert loaded.is_due(now - 1) is False and loaded.is_due(now) is True  # due exactly on the clock

    assert len(_fire(server, sid, s)) == 1  # the poller's heartbeat driver fires it
    after = load_heartbeat(key)
    assert after.fire_count == 1 and after.last_fired_at > now  # re-anchored past the stale slot
    assert _fire(server, sid, s) == [] and load_heartbeat(key).fire_count == 1

    # And the manager re-anchors exactly on an injected clock (no drift, no repeat fire).
    mgr2 = HeartbeatManager(key)
    anchor = after.last_fired_at + CADENCE
    assert mgr2.due_prompt(anchor - 1) is None  # not yet due at the injected instant
    assert mgr2.due_prompt(anchor) is not None  # due exactly at the anchor
    rearmed = load_heartbeat(key)
    assert rearmed.last_fired_at == anchor and rearmed.fire_count == 2
    assert rearmed.is_due(anchor) is False
