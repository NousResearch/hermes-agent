"""/heartbeat firing from the TUI/Desktop session-owner process (#102056, #103044).

The slash worker that parses ``/heartbeat`` runs a HermesCLI whose watchdog queues the due prompt
into its own ``_pending_input`` — a queue no turn loop drains in that process. The per-session
notification poller (the same driver that fires ``/loop``) must poll the persisted HeartbeatManager
and re-enter the live session through ``_run_prompt_submit``.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
    sid, key = "sid-hb-test", "tui-hb-session-1"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock()}
    server._sessions[sid] = s
    return sid, key, s


def _arm_due(key: str):
    from hermes_cli.heartbeat import HeartbeatManager, save_heartbeat

    mgr = HeartbeatManager(key)
    state = mgr.set("report backend health", 60)
    state.created_at = time.time() - 3600
    save_heartbeat(key, state)
    return mgr


def _submits(server, submit):
    return patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit")


def test_notification_poller_fires_due_heartbeat_when_idle(server, session):
    """The session-owner poller loop itself dispatches a due heartbeat exactly once; the same state on
    the base loop never fired (armed-but-dead)."""
    sid, key, s = session
    _arm_due(key)
    dispatched: list[str] = []

    def submit(rid, sid_, session_, text, **kw):
        dispatched.append(text)
        return True

    stop = threading.Event()
    p_submit, p_emit = _submits(server, submit)
    with p_submit, p_emit:
        t = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, s), daemon=True)
        t.start()
        deadline = time.monotonic() + 8
        while not dispatched and time.monotonic() < deadline:
            time.sleep(0.1)
        stop.set()
        t.join(timeout=5)

    from hermes_cli.heartbeat import load_heartbeat

    assert len(dispatched) == 1 and "report backend health" in dispatched[0]
    assert s["running"] is True  # claimed for the heartbeat turn
    assert load_heartbeat(key).fire_count == 1 and not load_heartbeat(key).is_due()


@pytest.mark.parametrize("running,due", [(True, True), (False, False)])
def test_heartbeat_tick_defers_when_busy_or_not_due(server, session, running, due):
    sid, key, s = session
    mgr = _arm_due(key)
    if not due:
        mgr.state.created_at = time.time()
        from hermes_cli.heartbeat import save_heartbeat

        save_heartbeat(key, mgr.state)
    s["running"] = running
    p_submit, p_emit = _submits(server, MagicMock())
    with p_submit as submit, p_emit:
        server._maybe_fire_tui_heartbeat_tick(sid, s)
    submit.assert_not_called()
    from hermes_cli.heartbeat import load_heartbeat

    assert s["running"] is running  # a busy session is never released by the poller
    assert load_heartbeat(key).fire_count == 0  # tick not consumed — still due when the session frees up


@pytest.mark.parametrize("failure", ["refused", "raised"])
def test_heartbeat_dispatch_that_never_starts_a_turn_stays_due(server, session, failure):
    """A refused/failed submit must release the claim AND rewind the persisted fire, otherwise the tick is
    silently consumed (fire_count advanced, nothing ran) — the Desktop-side follow-up bug in #104011."""
    sid, key, s = session
    _arm_due(key)

    def submit(*a, **k):
        if failure == "raised":
            raise RuntimeError("transport down")
        with s["history_lock"]:
            s["running"] = False  # _admit_prompt_turn refusal releases itself and returns False
        return False

    p_submit, p_emit = _submits(server, submit)
    with p_submit, p_emit:
        server._maybe_fire_tui_heartbeat_tick(sid, s)

    from hermes_cli.heartbeat import load_heartbeat

    assert s["running"] is False
    assert load_heartbeat(key).fire_count == 0 and load_heartbeat(key).is_due()


def test_abandon_fire_never_overwrites_a_concurrent_pause(hermes_home):
    from hermes_cli.heartbeat import HeartbeatManager, load_heartbeat

    key = "hb-abandon-race"
    driver = _arm_due(key)
    assert driver.due_prompt() is not None
    HeartbeatManager(key).pause()  # lands between the claim and the failed dispatch
    assert driver.abandon_fire() is False
    assert load_heartbeat(key).status == "paused" and load_heartbeat(key).fire_count == 1
