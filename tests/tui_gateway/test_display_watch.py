"""Lease transitions made by ANOTHER process reach `hermes serve` clients as ``display.lease``.

The agent lives in whatever process hosts it (messaging gateway, ``hermes chat``, a cron worker);
``lease.on_change`` is in-process only, so the serve backend must notice the file move itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _other_process(home: Path, stmt: str) -> None:
    env = {**os.environ, "HERMES_HOME": str(home)}
    subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO)!r}); "
         f"from tools.bot_desktop import lease; {stmt}"],
        cwd=str(REPO), env=env, check=True, timeout=60, stdin=subprocess.DEVNULL)


def _wait_for(pred, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def _serialize_polls(server, monkeypatch) -> None:
    """The process-wide watcher thread runs the same poll passes every 0.5 s that these tests drive by
    hand. A pass is check-mark → snapshot → broadcast, so a watcher pass that claims a transition first
    broadcasts it only after its (slow, under load) snapshot — past the test's assert, which saw an
    unchanged mark and no event. One lock around every pass (the thread resolves these names from the
    server globals on each tick) makes a hand-driven poll return only once any claimed transition is out."""
    lock = threading.Lock()
    for name in ("_poll_lease_files", "_poll_runtime_files"):
        def locked(real=getattr(server, name)) -> None:
            with lock:
                real()
        monkeypatch.setattr(server, name, locked)


def _watching(server, home: Path, monkeypatch) -> list:
    from hermes_constants import hermes_home_key
    events: list = []
    # Before the home is watched: no pass that can see it runs unlocked.
    _serialize_polls(server, monkeypatch)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload)))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    server._ensure_lease_watcher()
    assert _wait_for(lambda: hermes_home_key(home) in server._lease_epochs), "watcher never seeded the home"
    return events


def _lease_events(events, **want):
    return [p for ev, p in events if ev == "display.lease" and all(p["lease"].get(k) == v for k, v in want.items())]


def test_takeover_in_another_process_is_broadcast(tmp_path, monkeypatch):
    from tui_gateway import server
    from hermes_constants import hermes_home_key
    home = tmp_path / "home"
    home.mkdir()
    events = _watching(server, home, monkeypatch)

    _other_process(home, 'lease.acquire("cli-viewer", reason="probe")')

    assert _wait_for(lambda: _lease_events(events, reason="probe")), events
    (payload,) = _lease_events(events, reason="probe")
    assert payload["profile_key"] == hermes_home_key(home)


def test_release_in_another_process_is_broadcast_and_local_transition_not_duplicated(tmp_path, monkeypatch):
    from tui_gateway import server
    from tools.bot_desktop import lease
    home = tmp_path / "home"
    home.mkdir()
    events = _watching(server, home, monkeypatch)
    server._install_lease_listener()  # what display.status does: in-process transitions broadcast here

    lease.acquire("viewer-1", profile_key=str(home))
    assert _wait_for(lambda: _lease_events(events, holder="human"))
    before = len(events)
    server._poll_lease_files()  # the file moved too, but the watcher saw that epoch locally: no re-broadcast
    assert len(events) == before

    _other_process(home, 'lease.release("viewer-1")')

    assert _wait_for(lambda: _lease_events(events, holder="agent")), events
    assert len(_lease_events(events, holder="agent")) == 1


def test_screen_started_or_stopped_by_another_process_is_broadcast_as_status(tmp_path, monkeypatch):
    """A start/stop made by the CLI or gateway process must reach an open Desktop: the portal's
    status was otherwise one-shot (fetched once, then only lease events)."""
    from tui_gateway import server

    home = tmp_path / "home"
    (home / "bot-desktop").mkdir(parents=True)
    events = _watching(server, home, monkeypatch)
    server._poll_runtime_files()  # seed
    assert not [e for e in events if e[0] == "display.status"]

    # what runtime.start() publishes from another process: launcher.pid then env
    (home / "bot-desktop" / "launcher.pid").write_text("424242 1.5", encoding="utf-8")
    (home / "bot-desktop" / "env").write_text("DISPLAY=:77\n", encoding="utf-8")
    server._poll_runtime_files()
    statuses = [p for e, p in events if e == "display.status"]
    assert statuses and statuses[-1]["profile_key"] == str(home)

    (home / "bot-desktop" / "env").unlink()  # stop() from the other process
    server._poll_runtime_files()
    assert len([e for e in events if e[0] == "display.status"]) == 2
    assert [p for e, p in events if e == "display.status"][-1]["running"] is False


def test_launcher_dying_without_touching_its_files_is_broadcast_as_stopped(tmp_path, monkeypatch):
    """Xvnc/the launcher crashing leaves env and launcher.pid exactly as they were, so a watcher keyed
    on mtimes alone never told the Desktop the screen was gone. The mark must include liveness."""
    import psutil
    from tui_gateway import server

    home = tmp_path / "home"
    (home / "bot-desktop").mkdir(parents=True)
    launcher = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL)
    try:
        born = psutil.Process(launcher.pid).create_time()
        (home / "bot-desktop" / "launcher.pid").write_text(f"{launcher.pid} {born!r}", encoding="utf-8")
        (home / "bot-desktop" / "env").write_text("DISPLAY=:77\n", encoding="utf-8")
        events = _watching(server, home, monkeypatch)
        server._poll_runtime_files()  # seed: running
        server._poll_runtime_files()
        assert not [e for e in events if e[0] == "display.status"], "unchanged state was re-broadcast"

        launcher.kill()
        launcher.wait(5)
        server._poll_runtime_files()
        server._poll_runtime_files()
        statuses = [p for e, p in events if e == "display.status"]
        assert len(statuses) == 1 and statuses[0]["running"] is False, statuses
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(5)


def test_screen_start_and_stop_reach_the_models_turn_notes(tmp_path, monkeypatch):
    """#125830: a screen started/stopped mid-session must reach the model without waiting for a
    prompt rebuild — the watcher stages the Bot Screen line as a one-shot per-turn note on every
    live session of THAT profile. The staged note is consumed on the next turn
    (``agent/turn_context.py``), so only its presence here is asserted."""
    import psutil
    from tui_gateway import server

    home = tmp_path / "home"
    (home / "bot-desktop").mkdir(parents=True)
    events = _watching(server, home, monkeypatch)

    class _Agent:  # a plain object: MagicMock would auto-create the note attribute
        _gateway_turn_context_notes = ""

    agent, other = _Agent(), _Agent()
    server._sessions["ws-a"] = {"agent": agent, "profile_home": str(home), "transport": server._detached_ws_transport}
    other_home = tmp_path / "other"
    other_home.mkdir()
    server._sessions["ws-b"] = {"agent": other, "profile_home": str(other_home), "transport": server._detached_ws_transport}
    # the stub "launcher": a real live process with matching create_time, like runtime.start() records
    launcher = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL)
    try:
        server._poll_runtime_files()  # seed: stopped
        assert not getattr(agent, "_gateway_turn_context_notes", "")

        born = psutil.Process(launcher.pid).create_time()
        (home / "bot-desktop" / "launcher.pid").write_text(f"{launcher.pid} {born!r}", encoding="utf-8")
        (home / "bot-desktop" / "env").write_text("DISPLAY=:77\n", encoding="utf-8")
        server._poll_runtime_files()
        note = getattr(agent, "_gateway_turn_context_notes", "")
        assert "Bot Screen" in note and "display :77" in note and "RUNNING" in note, note
        assert not getattr(other, "_gateway_turn_context_notes", ""), "another profile's session was told"

        (home / "bot-desktop" / "env").unlink()  # stop() from another process
        server._poll_runtime_files()
        note = getattr(agent, "_gateway_turn_context_notes", "")
        assert "no longer running" in note, note
        assert [e for e in events if e[0] == "display.status"]  # the broadcast still happened
    finally:
        with server._sessions_lock:
            server._sessions.pop("ws-a", None)
            server._sessions.pop("ws-b", None)
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(5)
