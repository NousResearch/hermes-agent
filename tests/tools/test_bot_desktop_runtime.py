"""Bot Desktop runtime: thumbnail and display-number allocation invariants."""

from __future__ import annotations

import contextlib
import threading
import time
import sys
from pathlib import Path

import pytest

from tools.bot_desktop import runtime, thumbnail


@pytest.mark.parametrize("pm", sorted(runtime.PACKAGES))
def test_every_required_binary_maps_to_an_installed_package(pm):
    """Each binary the launcher execs must come from a package the distro list actually installs; dnf5
    refuses the whole transaction on one retired name, so the map is the contract, not the list."""
    mapping = runtime.BINARY_PACKAGES[pm]
    assert set(mapping) == set(runtime.REQUIRED_BINARIES)
    assert set(mapping.values()) <= set(runtime.PACKAGES[pm])
    assert not {"xorg-x11-server-utils", "xorg-x11-utils"} & set(runtime.PACKAGES["dnf"]), "retired on Fedora"


def test_no_running_screen_returns_none_without_grabbing(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":99"})
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)  # an import would now fail loudly
    assert thumbnail.thumbnail_data_url() is None


def test_recycled_pid_is_not_our_launcher(tmp_path, monkeypatch):
    """launcher.pid names pid + create_time; a live pid born at another time is a stranger (recycled pid)
    and must read as not running, or stop() would killpg an unrelated session. Legacy single-number
    files and absurd digit strings are also not running."""
    import os

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    pidfile = tmp_path / "launcher.pid"
    pidfile.write_text(f"{os.getpid()} 12345.0", encoding="utf-8")  # alive, wrong birth
    assert runtime._launcher_pid() is None
    pidfile.write_text(str(os.getpid()), encoding="utf-8")  # pre-identity format
    assert runtime._launcher_pid() is None
    pidfile.write_text("9" * 40 + " 1.0", encoding="utf-8")
    assert runtime._launcher_pid() is None
    pidfile.write_text(f"{os.getpid()} {runtime._create_time(os.getpid())}", encoding="utf-8")
    assert runtime._launcher_pid() == os.getpid()


def test_recorded_display_held_by_a_live_server_is_not_reused(tmp_path, monkeypatch):
    """After profile A stops, B may take A's number; A restarting must pick another rather than
    unlink B's socket and lock."""
    import os

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    (tmp_path / "display").write_text("37", encoding="utf-8")
    live = {37: os.getpid()}  # :37 is owned by a running server (this very process stands in for it)
    monkeypatch.setattr(runtime, "_display_in_use", lambda num: num in live)
    monkeypatch.setattr(runtime, "_ALLOC_LOCK", tmp_path / "alloc.lock")
    assert runtime._allocate_display() != 37
    live.clear()
    assert runtime._allocate_display() == 37, "a free recorded number is reclaimed"


def test_live_server_without_its_lock_file_still_owns_the_display(tmp_path, monkeypatch):
    """Regression for #109941: a /tmp reaper removes ``.X<n>-lock`` while Xvnc keeps running. The server's
    abstract socket ``@/tmp/.X11-unix/X<n>`` stays bound for its whole life, so that is the liveness the
    allocator must honour — handing the number out makes the next launcher die with 'server already running'."""
    monkeypatch.setattr(runtime, "_X_LOCK_DIR", tmp_path / "xlocks")  # no lock file for anyone
    (tmp_path / "xlocks").mkdir()
    table = tmp_path / "unix"
    table.write_text(
        "Num       RefCount Protocol Flags    Type St Inode Path\n"
        "0000000000000000: 00000002 00000000 00010000 0001 01 22242 @/tmp/.X11-unix/X20\n"
        "0000000000000000: 00000002 00000000 00010000 0001 01 22243 /tmp/.X11-unix/X21\n"
        "0000000000000000: 00000003 00000000 00000000 0001 03 22244 @/tmp/.X11-unix/X200\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_X_UNIX_TABLE", table)
    assert runtime._display_in_use(20) is True
    assert runtime._display_in_use(21) is True
    assert runtime._display_in_use(22) is False  # :200 is not :22 — no prefix matching
    monkeypatch.setattr(runtime, "_X_UNIX_TABLE", tmp_path / "missing")
    assert runtime._display_in_use(20) is False, "no table (non-Linux procfs) falls back to the lock alone"



_FAKE_LAUNCHER = """#!/usr/bin/env bash
# Stands in for launcher.sh + Xvnc: the X lock appears only after a delay (the TOCTOU window), then the
# env file + socket are published; stays alive until killed like the real supervisor.
: > "$HERMES_BD_XLOCK_DIR/spawned.$$"
sleep 0.4
echo $$ > "$HERMES_BD_XLOCK_DIR/.X${HERMES_BD_DISPLAY_NUM}-lock"
: > "$HERMES_BD_SOCKET"
printf 'DISPLAY=:%s\\n' "$HERMES_BD_DISPLAY_NUM" > "$HERMES_BD_ENV_FILE"
sleep 30
"""

# One start() per process: state_dir() is HERMES_HOME-scoped and process-global, so two profiles need two
# interpreters — which is also how two gateway profiles race on a real host.
_DRIVER = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
from tools.bot_desktop import runtime
scratch = Path({scratch!r})
runtime._LAUNCHER = scratch / "launcher.sh"
runtime._X_LOCK_DIR = scratch / "xlocks"
runtime._ALLOC_LOCK = scratch / "alloc.lock"
runtime.missing_binaries = lambda: []
runtime.geometry = lambda: "800x600"
os.environ["HERMES_BD_XLOCK_DIR"] = str(scratch / "xlocks")
try:
    st = runtime.start(wait_seconds=10)
    print(json.dumps({{"display": st.display, "pid": st.pid}}), flush=True)
except Exception as exc:
    print(json.dumps({{"error": str(exc)}}), flush=True)
sys.stdin.readline()  # the test releases us once every driver has reported; we own the launcher, we stop it
runtime.stop()
"""


@pytest.fixture
def start_in_fresh_process(tmp_path):
    import os
    import subprocess

    (tmp_path / "launcher.sh").write_text(_FAKE_LAUNCHER, encoding="utf-8")
    (tmp_path / "xlocks").mkdir()
    repo = str(Path(__file__).resolve().parents[2])
    procs: list[subprocess.Popen] = []

    def launch(home: Path) -> subprocess.Popen:
        env = {**os.environ, "HERMES_HOME": str(home)}
        proc = subprocess.Popen([sys.executable, "-c", _DRIVER.format(repo=repo, scratch=str(tmp_path))],
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        return proc

    yield launch
    for proc in procs:
        with contextlib.suppress(OSError):
            proc.communicate("go\n", timeout=20)
        proc.kill()


def _collect(procs):
    import json
    out = [json.loads(p.stdout.readline()) for p in procs]  # every driver holds its launcher until released
    assert all("error" not in o for o in out), out
    return out


@pytest.mark.linux_only
def test_concurrent_cold_starts_of_two_profiles_get_distinct_displays(tmp_path, start_in_fresh_process):
    """The allocation lock must outlive the pick: Xvnc writes /tmp/.X<n>-lock well after start() chose n, so
    a second profile starting in that window used to pick the same n (and its launcher's stale-lock cleanup
    could then unlink the winner's socket)."""
    out = _collect([start_in_fresh_process(tmp_path / "a"), start_in_fresh_process(tmp_path / "b")])
    assert len({o["display"] for o in out}) == 2, out


@pytest.mark.linux_only
def test_concurrent_starts_of_one_profile_spawn_one_launcher(tmp_path, start_in_fresh_process):
    """Two start() calls for one profile spawn ONE launcher; the second used to spawn its own, overwrite
    launcher.pid and orphan the first (both callers then reported the last-written pid)."""
    out = _collect([start_in_fresh_process(tmp_path / "a"), start_in_fresh_process(tmp_path / "a")])
    assert len({o["pid"] for o in out}) == 1, out
    assert len(list((tmp_path / "xlocks").glob("spawned.*"))) == 1


_ORPHANING_LAUNCHER = """#!/usr/bin/env bash
# Stands in for launcher.sh whose Xvnc child ("sleep") lives in the launcher's process group and
# outlives a SIGKILL of the launcher itself — the X lock names the child, as the real one does.
: > "$HERMES_BD_XLOCK_DIR/spawned.$$"
sleep 30 &
echo $! > "$HERMES_BD_XLOCK_DIR/.X${HERMES_BD_DISPLAY_NUM}-lock"
: > "$HERMES_BD_SOCKET"
printf 'DISPLAY=:%s\\n' "$HERMES_BD_DISPLAY_NUM" > "$HERMES_BD_ENV_FILE"
wait
"""


def _gone(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_until(pred, timeout=5.0) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


@pytest.fixture
def in_process_runtime(tmp_path, monkeypatch):
    """runtime.start()/stop() against a scratch state dir and a fake launcher script (set by the test)."""
    import os

    home = tmp_path / "home"
    (tmp_path / "xlocks").mkdir()
    monkeypatch.setattr(runtime, "state_dir", lambda: home / "bot-desktop")
    monkeypatch.setattr(runtime, "_LAUNCHER", tmp_path / "launcher.sh")
    monkeypatch.setattr(runtime, "_X_LOCK_DIR", tmp_path / "xlocks")
    monkeypatch.setattr(runtime, "_X_UNIX_TABLE", tmp_path / "unix")  # the host's real X servers stay out of the band
    monkeypatch.setattr(runtime, "_ALLOC_LOCK", tmp_path / "alloc.lock")
    monkeypatch.setattr(runtime, "missing_binaries", lambda: [])
    monkeypatch.setattr(runtime, "geometry", lambda: "800x600")
    monkeypatch.setenv("HERMES_BD_XLOCK_DIR", str(tmp_path / "xlocks"))
    yield tmp_path
    with contextlib.suppress(Exception):
        runtime.stop()
    for lock in (tmp_path / "xlocks").glob(".X*-lock"):  # anything the code under test failed to reap
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(lock.read_text()), 9)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass  # the orphan is reparented to init: signalling it is the point
def test_orphaned_x_server_of_a_dead_launcher_is_reaped_on_next_start(in_process_runtime):
    """SIGKILL the launcher and its Xvnc survives, holding the display and rfb.sock. status() keys on the
    launcher pid and says stopped; start() must find that orphan through the recorded display's X lock
    and kill it instead of allocating a second server beside it (two servers, one socket path)."""
    import os
    import signal

    scratch = in_process_runtime
    (scratch / "launcher.sh").write_text(_ORPHANING_LAUNCHER, encoding="utf-8")
    first = runtime.start(wait_seconds=10)
    lock = scratch / "xlocks" / f".X{first.display.lstrip(':')}-lock"
    orphan = int(lock.read_text())
    os.kill(first.pid, signal.SIGKILL)
    assert _wait_until(lambda: _gone(first.pid))
    assert not _gone(orphan), "the X server outlives its launcher (that is the bug's precondition)"
    assert runtime.status().running is False

    second = runtime.start(wait_seconds=10)
    assert second.pid != first.pid and second.running
    assert _wait_until(lambda: _gone(orphan)), "the dead launcher's X server must be reaped, not leaked"
    assert runtime.stop() is True


_SLOW_LAUNCHER = """#!/usr/bin/env bash
# Publishes only AFTER runtime.start()'s readiness deadline has passed.
sleep 30 &
echo $! > "$HERMES_BD_XLOCK_DIR/.X${HERMES_BD_DISPLAY_NUM}-lock"
sleep 1
: > "$HERMES_BD_SOCKET"
printf 'DISPLAY=:%s\\n' "$HERMES_BD_DISPLAY_NUM" > "$HERMES_BD_ENV_FILE"
wait
"""


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass  # the launcher's group must really be signalled
def test_readiness_timeout_terminates_the_launch_it_gave_up_on(in_process_runtime):
    """When the launcher misses the readiness deadline start() raises — and must take the launch down with
    it. It used to leave the launcher running; the child then published DISPLAY/rfb.sock a moment later and
    a screen nobody asked for (and whose start() had reported failure) stayed up behind a 'running' status."""
    import time

    scratch = in_process_runtime
    (scratch / "launcher.sh").write_text(_SLOW_LAUNCHER, encoding="utf-8")
    sd = runtime.state_dir()
    with pytest.raises(RuntimeError, match="did not publish"):
        runtime.start(wait_seconds=0.05)
    launcher = runtime._recorded_launcher_pid()
    assert launcher is None or _gone(launcher), "the timed-out launcher must be reaped, not left to publish later"
    time.sleep(1.5)  # past the slow launcher's publish time
    assert not (sd / "env").exists() and not (sd / "rfb.sock").exists()
    assert runtime.status().running is False
    for lock in (scratch / "xlocks").glob(".X*-lock"):
        assert _gone(int(lock.read_text())), "the launch's X server must die with its launcher"


_DYING_LAUNCHER = """#!/usr/bin/env bash
# Xvnc refusing the number ('server already running'): the launcher exits non-zero at once.
exit 1
"""


@pytest.mark.linux_only
def test_failed_start_does_not_pin_the_profile_to_the_number_that_failed(in_process_runtime):
    """Regression for #109941: after a launcher failure the recorded ``display`` kept naming the number, and
    ``_pick_display`` reuses the recorded number first — so every retry picked the same occupied display and
    the profile wedged. A failed start forgets its number; the next attempt allocates afresh."""
    scratch = in_process_runtime
    (scratch / "launcher.sh").write_text(_DYING_LAUNCHER, encoding="utf-8")
    sd = runtime.state_dir()
    with pytest.raises(RuntimeError, match="launcher exited"):
        runtime.start(wait_seconds=5)
    assert not (sd / "display").exists(), "a number that just failed must not be recorded for reuse"


@pytest.mark.linux_only
def test_allocation_lock_is_released_once_xvnc_claims_the_number(in_process_runtime):
    """The host-wide allocation lock exists for the pick→X-lock window only. Holding it for the whole Xfce
    bring-up serialized every profile's start behind one desktop launch (and a hung launcher blocked them
    all for the full timeout): once /tmp/.X<n>-lock exists the number is Xvnc's and the lock must be free."""
    import fcntl

    scratch = in_process_runtime
    # X lock at once, env file only much later: the lock must be free in between.
    (scratch / "launcher.sh").write_text(
        '#!/usr/bin/env bash\necho $$ > "$HERMES_BD_XLOCK_DIR/.X${HERMES_BD_DISPLAY_NUM}-lock"\n'
        'sleep 1.5\n: > "$HERMES_BD_SOCKET"\nprintf \'DISPLAY=:%s\\n\' "$HERMES_BD_DISPLAY_NUM" > "$HERMES_BD_ENV_FILE"\nsleep 30\n',
        encoding="utf-8")
    seen: dict = {}

    def probe():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if list((scratch / "xlocks").glob(".X*-lock")):
                time.sleep(0.2)  # let start() notice the claim
                with open(scratch / "alloc.lock", "a+") as fh:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        seen["free"] = True
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        seen["free"] = False
                return
            time.sleep(0.05)

    t = threading.Thread(target=probe)
    t.start()
    st = runtime.start(wait_seconds=10)
    t.join()
    assert st.running
    assert seen.get("free") is True, "allocation lock still held after Xvnc wrote its X lock"
