"""A Bot Desktop screen whose gateway never comes back stops itself after ``bot_desktop.idle_stop_minutes``.

The real ``launcher.sh`` runs against stub X/Xfce binaries. ``runtime.start()`` runs in a separate
"gateway" process that is SIGKILLed, so nothing in the test process holds the screen.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.waiting import wait_until

pytestmark = [pytest.mark.platforms("linux"), pytest.mark.live_system_guard_bypass]

_REPO = Path(__file__).resolve().parents[2]

_STUBS = {
    # Binds nothing: creates the socket path it was given, then lives like a server.
    "Xvnc": '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -rfbunixpath ] && : > "$2"; shift; done\nexec sleep 600\n',
    "dbus-run-session": '#!/bin/sh\nshift\nDBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent; export DBUS_SESSION_BUS_ADDRESS\nexec "$@"\n',
    "xfce4-panel": "#!/bin/sh\nexec sleep 600\n",
    **{name: "#!/bin/sh\nexit 0\n" for name in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "xauth", "xprop",
                                                  "xfsettingsd", "xfwm4", "xfdesktop")},
}

# The gateway: starts the profile's screen through the real runtime path, reports the launcher pid, idles.
_GATEWAY = """
import sys, time
from pathlib import Path
sys.path.insert(0, {repo!r})
from tools.bot_desktop import runtime
runtime._X_LOCK_DIR = Path({scratch!r})
runtime._ALLOC_LOCK = runtime._X_LOCK_DIR / "alloc.lock"
runtime.missing_binaries = list
{prelude}
print(runtime.start(wait_seconds=15).pid, flush=True)
time.sleep(600)
"""


def _alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)  # windows-footgun: ok — Linux-only test
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def abandoned_screen(tmp_path):
    """``launch(idle_stop_minutes)`` -> (state dir, launcher pid): a running screen whose gateway is dead."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in _STUBS.items():
        (stubs / name).write_text(body, encoding="utf-8")
        (stubs / name).chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}
    env.pop("PYTHONPATH", None)
    groups: list[int] = []

    def launch(idle_stop_minutes: float, prelude: str = "") -> tuple[Path, int]:
        (home / "config.yaml").write_text(f"bot_desktop:\n  idle_stop_minutes: {idle_stop_minutes}\n", encoding="utf-8")
        gateway = subprocess.Popen([sys.executable, "-c", _GATEWAY.format(repo=str(_REPO), scratch=str(tmp_path), prelude=prelude)],
                                   env=env, stdout=subprocess.PIPE, text=True)
        launcher = int(gateway.stdout.readline())
        groups.append(launcher)
        gateway.kill()  # SIGKILL: no cleanup runs, no gateway comes back
        gateway.wait()
        gateway.stdout.close()
        return home / "bot-desktop", launcher

    yield launch
    for pgid in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — Linux-only test


def test_an_abandoned_screen_stops_itself_after_the_idle_window(abandoned_screen):
    sd, launcher = abandoned_screen(idle_stop_minutes=0.02)  # 1.2 s
    # stop() deletes its runtime files after the group is gone, so they are part of "stopped".
    files = ("launcher.pid", "env", "activity", "idle-watch.out")
    wait_until(lambda: not _alive(launcher) and not any((sd / name).exists() for name in files),
               timeout=30, what="the abandoned screen to be stopped the way stop() does")


def test_a_screen_started_before_an_update_still_stops_itself(abandoned_screen, tmp_path):
    """The screen outlives ``hermes update``, which can drop the starter's interpreter and collect the dependency
    generation its PYTHONPATH named: a check pinned to either failed with ModuleNotFoundError and was retried in
    300 s, forever. The launcher runs the install's stable command (on a store install the ``hermes
    --run-module`` launcher, which loads the current generation at every run) with its own pid appended."""
    install = tmp_path / "an install's root"  # quoting: the command crosses the env as a shell-quoted string
    used = tmp_path / "launcher-used"
    hermes = install / ".hermes" / "bin" / "hermes"
    hermes.parent.mkdir(parents=True)
    hermes.write_text(
        "#!/bin/sh\n"
        '[ "$1" = --run-module ] && [ "$2" = tools.bot_desktop.idle_watch ] && [ $# -eq 3 ] || exit 64\n'
        f'echo "$3" > "{used}"\n'
        "shift 2\n"
        f'PYTHONPATH="{_REPO}" exec "{sys.executable}" -m tools.bot_desktop.idle_watch "$@"\n', encoding="utf-8")
    hermes.chmod(0o755)
    prelude = (
        "from hermes_cli import _launchers\n"
        f"_launchers.resolve_store_python = lambda root, **_: Path({str(tmp_path / 'store/bin/python3')!r})\n"
        f"runtime.__file__ = {str(install / 'tools/bot_desktop/runtime.py')!r}\n"
        f"sys.executable = {str(tmp_path / 'old-env/bin/python')!r}  # removed by the update\n"
        f"import os; os.environ['PYTHONPATH'] = {str(tmp_path / 'collected-generation')!r}")
    sd, launcher = abandoned_screen(idle_stop_minutes=0.02, prelude=prelude)
    wait_until(lambda: not _alive(launcher) and not (sd / "launcher.pid").exists(), timeout=30,
               what="the abandoned screen to stop itself through the install's launcher")
    assert used.read_text(encoding="utf-8").split() == [str(launcher)]


def test_the_first_idle_check_waits_until_the_screen_could_be_due(abandoned_screen):
    """A screen just started cannot be idle: checking at launch costs an interpreter start and races Xfce
    coming up, so the launcher sleeps a window before its first check."""
    started = time.monotonic()
    sd, _ = abandoned_screen(idle_stop_minutes=0.15)  # 9 s
    log = sd / "launcher.log"

    def checked_while_used() -> bool:
        (sd / "activity").touch()  # in use: the first check keeps it
        return "idle-watch" in log.read_text(encoding="utf-8", errors="replace")

    wait_until(checked_while_used, timeout=30, what="the launcher's first idle check")
    assert time.monotonic() - started >= 7, "the first idle check ran at launch"


def test_a_self_stop_does_not_sit_out_the_kill_grace(abandoned_screen):
    """The check's stop() TERMs the launcher's group, which holds the loop that ran the check. A loop blocked
    in ``$(check)`` defers its TERM trap until the check returns, while the check waits for the group to
    empty: every self-stop sat out stop()'s 5 s grace and ended in SIGKILL, with start.lock held throughout."""
    sd, launcher = abandoned_screen(idle_stop_minutes=0.02)
    log = sd / "launcher.log"
    wait_until(lambda: "stopping" in log.read_text(encoding="utf-8", errors="replace"), timeout=30,
               what="the idle check to start stopping the screen")
    began = time.monotonic()
    wait_until(lambda: not _alive(launcher), timeout=30, what="the screen's process group to be gone")
    assert time.monotonic() - began < 3, "the self-stop waited for stop()'s SIGKILL grace"


def test_a_killed_sleep_does_not_end_the_self_stop(abandoned_screen):
    """The watch loop ran under the launcher's ``set -e``: a sleep killed from outside made ``wait`` fail and
    ended the loop silently, so that screen never stopped itself."""
    import psutil
    _, launcher = abandoned_screen(idle_stop_minutes=0.15)  # the first check is due 9 s after start

    def first_sleep():
        for proc in psutil.process_iter(["cmdline"]):
            with contextlib.suppress(psutil.Error):
                if proc.info["cmdline"] == ["sleep", "9"] and os.getpgid(proc.pid) == launcher:
                    return proc
        return None

    wait_until(first_sleep, timeout=10, what="the watch loop's first sleep").kill()
    wait_until(lambda: not _alive(launcher), timeout=60, what="the screen to stop itself after its sleep was killed")


def test_a_screen_in_use_is_kept_and_a_returning_gateway_reattaches(abandoned_screen, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import runtime

    sd, launcher = abandoned_screen(idle_stop_minutes=0.1)  # 6 s; a human lease lapses after 10 min (the floor over 4 windows)
    log = sd / "launcher.log"

    def checks_kept() -> int:
        return log.read_text(encoding="utf-8", errors="replace").count("screen kept")

    def kept_while_used(n: int) -> bool:
        (sd / "activity").touch()  # the agent / a viewer using the screen
        return checks_kept() >= n

    wait_until(lambda: kept_while_used(2), timeout=30, what="idle checks that keep a screen in use")
    assert _alive(launcher)

    # A gateway that comes back inside the window re-attaches to the same screen instead of starting another.
    monkeypatch.setattr(runtime, "missing_binaries", list)  # the stub X binaries are only on the launcher's PATH
    token = set_hermes_home_override(sd.parent)
    try:
        assert runtime.start(wait_seconds=1).pid == launcher
    finally:
        reset_hermes_home_override(token)

    # A human holding the screen past the window is not timed out... Synced to a check that just kept the screen
    # in use, so the next one starts a window later and judges the human lease, not the use before it.
    seen = checks_kept()
    wait_until(lambda: kept_while_used(seen + 1), timeout=30, what="an idle check to sync on")
    (sd / "lease.json").write_text(json.dumps({"holder": "human", "viewer_id": "v", "epoch": 1}), encoding="utf-8")
    old = time.time() - 6.2
    os.utime(sd / "activity", (old, old))
    seen = checks_kept()
    wait_until(lambda: checks_kept() >= seen + 1, timeout=30, what="an idle check while a human holds the screen")
    assert _alive(launcher)

    # ...but a human lease nothing has used past its lapse (the viewer's link dropped and never came back)
    # no longer keeps it.
    old = time.time() - 3600
    os.utime(sd / "activity", (old, old))
    wait_until(lambda: not _alive(launcher), timeout=30, what="the screen to stop once its human lease is abandoned")
    # ...and the check that stopped it hands the abandoned lease back, so the next use is not fenced out.
    wait_until(lambda: json.loads((sd / "lease.json").read_text(encoding="utf-8"))["holder"] == "agent", timeout=10,
               what="the abandoned human lease to be handed back to the agent")
