"""Bot Desktop idle auto-stop: a screen nobody used past ``bot_desktop.idle_stop_minutes`` is stopped,
one a human holds or that saw recent use is not."""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time

import pytest

from tools.bot_desktop import lease, runtime


def _running(monkeypatch, tmp_path):
    sd = tmp_path / "bd"
    sd.mkdir()
    (sd / "env").write_text("DISPLAY=:20\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "state_dir", lambda: sd)
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: 4242)
    monkeypatch.setattr(runtime, "idle_stop_seconds", lambda: 600.0)
    stops = []
    monkeypatch.setattr(runtime, "stop", lambda still_due=None: stops.append(True) or True)
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

    _age(sd / "activity", 1200)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.HUMAN, viewer_id="v1"))
    assert runtime.stop_if_idle() is False and stops == [], "a human mid-login is never timed out"

    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.AGENT))
    assert runtime.stop_if_idle() is True and stops == [True]


def test_a_human_lease_keeps_the_screen_until_nothing_has_used_it_for_four_windows(tmp_path, monkeypatch):
    """A viewer link that dropped (laptop lid, a dead gateway) keeps the human's lease forever, and a corrupt
    lease reads as human: with nobody coming back the screen must still stop eventually."""
    sd, stops = _running(monkeypatch, tmp_path)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.HUMAN, viewer_id="v1"))
    (sd / "activity").touch()
    windows = runtime.HUMAN_LEASE_ABANDONED_WINDOWS

    _age(sd / "activity", 600 * windows - 60)
    assert runtime.stop_if_idle() is False and stops == [], "kept: a human may still be at it"

    _age(sd / "activity", 600 * windows + 60)
    assert runtime.stop_if_idle() is True and stops == [True], "abandoned: no use for four windows"


def test_idle_stop_disabled_by_zero(tmp_path, monkeypatch):
    sd, stops = _running(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "idle_stop_seconds", lambda: 0.0)
    _age(sd / "env", 10 ** 6)
    assert runtime.stop_if_idle() is False and stops == []


def test_the_launchers_idle_check_sleeps_until_the_screen_could_be_due():
    from tools.bot_desktop.idle_watch import next_check_in
    assert next_check_in(1800, 600) == 1200  # wakes when the window runs out at the current age
    assert next_check_in(1800, 1795) == 30  # never a tight loop near the edge
    assert next_check_in(1800, 7200) == 1800, "a human holding it past the window: recheck a window later"
    assert next_check_in(1.2, None) == 2  # a window shorter than the floor is checked at its own length
    assert next_check_in(0, None) == runtime.DEFAULT_IDLE_STOP_MINUTES * 60  # off: re-read the setting later


@pytest.mark.parametrize("judged", ["the old screen", "any screen"])
def test_an_idle_check_that_waited_on_the_lock_never_stops_the_replacement_screen(tmp_path, monkeypatch, judged):
    """The idle decision is taken before ``start.lock``; a stop + start that holds the lock meanwhile replaces
    the screen. The waiting check must re-decide under the lock: the launcher's own check (it names the screen
    it judged) keeps a replacement however idle, the gateway watcher keeps one that is in use."""
    from tools.bot_desktop import placement, sandbox_host

    sd = tmp_path / "bd"
    sd.mkdir()
    (sd / "env").write_text("DISPLAY=:20\n", encoding="utf-8")
    (sd / "activity").touch()
    _age(sd / "activity", 3600)
    monkeypatch.setattr(runtime, "state_dir", lambda: sd)
    monkeypatch.setattr(runtime, "idle_stop_seconds", lambda: 600.0)
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(sandbox_host, "_read_marker", dict)
    monkeypatch.setattr(placement, "resolve", lambda: placement.Placement(placement.GATEWAY, "local"))
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.AGENT))

    def launch() -> subprocess.Popen:
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        (sd / "launcher.pid").write_text(f"{proc.pid} {runtime._create_time(proc.pid)}", encoding="utf-8")
        return proc

    old, new = launch(), None
    lock, waiting, result = runtime._flocked, threading.Event(), []

    def flocked_after_signalling(path):
        waiting.set()
        return lock(path)
    monkeypatch.setattr(runtime, "_flocked", flocked_after_signalling)
    identity = (old.pid, runtime._create_time(old.pid)) if judged == "the old screen" else None
    check = threading.Thread(target=lambda: result.append(runtime.stop_if_idle(launcher=identity)))
    try:
        with lock(sd / "start.lock"):  # a concurrent stop + start holds it
            check.start()
            assert waiting.wait(5)  # past its first decision; it gets the lock only after the swap below
            old.kill()
            old.wait()
            new = launch()
            if judged == "any screen":
                runtime.touch_activity()  # the replacement is in use
        check.join(15)
        assert result == [False], "the waiting check stopped the screen it found under the lock"
        assert new.poll() is None and runtime._launcher_pid() == new.pid
    finally:
        for proc in (old, new):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()


def _spawn(monkeypatch, tmp_path) -> dict[str, str]:
    """Runs ``_spawn_and_wait`` up to the launcher's Popen (a launcher that exits at once); the env it got."""
    from tools.bot_desktop import browser

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(runtime, "_ALLOC_LOCK", tmp_path / "alloc.lock")
    monkeypatch.setattr(runtime, "_pick_display", lambda: 20)
    monkeypatch.setattr(browser, "dock_launch", lambda: None)
    handed = []

    class _Exited:
        pid, returncode = os.getpid(), 1

        def poll(self):
            return 1

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda argv, env, **_: handed.append(env) or _Exited())
    with pytest.raises(RuntimeError):
        runtime._spawn_and_wait(tmp_path, 1)
    return handed[0]


@pytest.mark.platforms("posix")
def test_the_launcher_runs_its_idle_check_through_the_installs_stable_command(tmp_path, monkeypatch):
    """The screen outlives ``hermes update``: a check pinned to the starter's interpreter and its inherited
    dependency generation fails with ModuleNotFoundError once that generation is collected, forever. On a store
    install the launcher gets the ``hermes --run-module`` launcher, which loads the current one at every run."""
    from hermes_cli import _launchers

    root = runtime._LAUNCHER.resolve().parents[2]
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda repo_root, **_: tmp_path / "store/bin/python3")

    command = shlex.split(_spawn(monkeypatch, tmp_path)["HERMES_BD_IDLE_CMD"])
    assert command == _launchers.installation_command(root, module="tools.bot_desktop.idle_watch")
    assert command == [str(root / ".hermes/bin/hermes"), "--run-module", "tools.bot_desktop.idle_watch"]


@pytest.mark.platforms("posix")
def test_a_stale_idle_check_never_writes_into_the_replacement_screens_log(tmp_path, monkeypatch):
    """The old screen's idle check runs in a session of its own and can outlive its screen (it waited on
    ``start.lock`` while a replacement started) with the old ``launcher.log`` open at its offset. A start that
    truncated the same file let that late write leave a run of NUL bytes and a "screen kept" line in the
    replacement's log, reading as if it were about the new screen."""
    log = tmp_path / "launcher.log"
    log.write_bytes(b"x" * 4096)
    with open(log, "r+b") as stale:  # the old check's stderr: the launcher's, not opened for append
        stale.seek(0, os.SEEK_END)
        _spawn(monkeypatch, tmp_path)
        stale.write(b"idle-watch changed while the idle stop waited; kept\n")
    assert log.read_bytes() == b""
