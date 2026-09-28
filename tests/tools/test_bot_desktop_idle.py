"""Bot Desktop idle auto-stop: a screen nobody used past ``bot_desktop.idle_stop_minutes`` is stopped,
one a human holds or that saw recent use is not."""

from __future__ import annotations

import os
import time

from tools.bot_desktop import lease, runtime


def _running(monkeypatch, tmp_path):
    sd = tmp_path / "bd"
    sd.mkdir()
    (sd / "env").write_text("DISPLAY=:20\n")
    monkeypatch.setattr(runtime, "state_dir", lambda: sd)
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: 4242)
    monkeypatch.setattr(runtime, "idle_stop_seconds", lambda: 600.0)
    stops = []
    monkeypatch.setattr(runtime, "stop", lambda: stops.append(True) or True)
    return sd, stops


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_idle_screen_is_stopped_but_a_used_or_held_one_is_not(tmp_path, monkeypatch):
    sd, stops = _running(monkeypatch, tmp_path)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.AGENT))

    _age(sd / "env", 5)  # just started, never stamped: ages from publish
    assert runtime.stop_if_idle() is False and stops == []

    _age(sd / "env", 3600)
    runtime.touch_activity()  # a computer_use action / browser spawn a moment ago
    assert runtime.stop_if_idle() is False and stops == []

    _age(sd / "activity", 3600)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.HUMAN, viewer_id="v1"))
    assert runtime.stop_if_idle() is False and stops == [], "a human mid-login is never timed out"

    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.AGENT))
    assert runtime.stop_if_idle() is True and stops == [True]


def test_idle_stop_disabled_by_zero(tmp_path, monkeypatch):
    sd, stops = _running(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "idle_stop_seconds", lambda: 0.0)
    _age(sd / "env", 10 ** 6)
    assert runtime.stop_if_idle() is False and stops == []
