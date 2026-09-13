"""Lease transitions made by ANOTHER process reach `hermes serve` clients as ``display.lease``.

The agent lives in whatever process hosts it (messaging gateway, ``hermes chat``, a cron worker);
``lease.on_change`` is in-process only, so the serve backend must notice the file move itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _other_process(home: Path, stmt: str) -> None:
    env = {**os.environ, "HERMES_HOME": str(home)}
    subprocess.run(  # noqa: S603
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


def _watching(server, home: Path, monkeypatch) -> list:
    from hermes_constants import hermes_home_key
    events: list = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload)))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    server._ensure_lease_watcher()
    assert _wait_for(lambda: hermes_home_key(home) in server._lease_epochs), "watcher never seeded the home"
    return events


def _lease_events(events, **want):
    return [p for ev, p in events if ev == "display.lease" and all(p["lease"].get(k) == v for k, v in want.items())]


def test_handoff_requested_in_another_process_is_broadcast(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from hermes_constants import hermes_home_key
    home = tmp_path / "home"
    home.mkdir()
    events = _watching(server, home, monkeypatch)

    _other_process(home, 'lease.request_handoff("probe")')

    assert _wait_for(lambda: _lease_events(events, pending_handoff="probe")), events
    (payload,) = _lease_events(events, pending_handoff="probe")
    assert payload["profile_key"] == hermes_home_key(home)


def test_release_in_another_process_is_broadcast_and_local_transition_not_duplicated(tmp_path, monkeypatch):
    import tui_gateway.server as server
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
