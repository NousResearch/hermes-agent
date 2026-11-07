"""A shared-profile Chromium held from an environment this process cannot verify (a container gateway's browser
seen from the host CLI, or the other way round) is never launched over, deleted or reaped from here.

Its ``SingletonLock`` names a pid of ANOTHER PID namespace (written under another hostname, or by a launch record
from another namespace), so the number proves nothing here. When its DevTools endpoint answers, Hermes attaches
and leases it as usual; when it does not, the launch fails with an error naming the profile instead of starting a
second Chromium on the same user-data-dir (which would wipe the running browser's ``DevToolsActivePort`` and fight
it for the profile).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = pytest.mark.platforms("linux")

_FOREIGN_NS = "4026531836:00000000-0000-0000-0000-000000000000"
_OTHER_HOST = "container-3f9c2a"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _relock(udd, host: str, pid: int) -> None:
    """Point the profile's SingletonLock at ``<host>-<pid>``: how Chromium in another environment wrote it."""
    lock = udd / "SingletonLock"
    lock.unlink(missing_ok=True)
    os.symlink(f"{host}-{pid}", lock)


def _foreign_launch_record(udd, pid: int) -> None:
    """The launch record a Hermes in another PID namespace wrote for ``pid`` (its own number for the browser)."""
    state = launcher._shared_state_dir(str(udd))
    name = f"{launcher._profile_digest(str(udd))[:16]}-{pid}-nffffffffffff.json"
    (state / name).write_text(json.dumps({
        "owner_pid": pid + 1, "owner_start_time": 1, "pid": pid, "start_time": 7, "user_data_dir": str(udd),
        "home_key": "elsewhere", "ephemeral": False, "ns": _FOREIGN_NS}))


def _start_fake(script, udd) -> subprocess.Popen:
    proc = subprocess.Popen([str(script), "--remote-debugging-port=0", f"--user-data-dir={udd}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_until(lambda: (udd / "DevToolsActivePort").exists(), timeout=15, what="the fake to come up")
    return proc


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_reachable_browser_from_another_environment_is_attached_and_never_stopped(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    theirs = _start_fake(script, udd)
    try:
        _relock(udd, _OTHER_HOST, _dead_pid())  # its pid in the container names nobody here
        port_file = (udd / "DevToolsActivePort").read_text()

        browser, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                ephemeral=False, headed=False, env=dict(os.environ), timeout=30)

        assert err is None and browser is not None and not browser.owned
        port, path = port_file.split()
        assert browser.cdp_url == f"ws://127.0.0.1:{port}{path}"
        assert pids(udd, "spawns") == [theirs.pid], "a second Chromium was started on the held profile"
        assert (udd / "DevToolsActivePort").read_text() == port_file
        state = launcher._shared_state_dir(str(udd))
        assert list(state.glob("*.lease")), "attached without a lease"
        assert launcher.get_browser(key) is browser

        launcher.stop_browser(key)
        assert launcher.reap_orphaned_browsers() == 0
        assert theirs.poll() is None, "stopped a browser another environment holds"
        assert not list(state.glob("*.lease"))
        assert (udd / "DevToolsActivePort").read_text() == port_file
    finally:
        kill_fake(theirs.pid, script)
        theirs.wait(timeout=10)


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_an_attached_browser_from_another_environment_is_dropped_once_its_endpoint_is_gone(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    theirs = _start_fake(script, udd)
    try:
        _relock(udd, _OTHER_HOST, _dead_pid())
        browser, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
        assert err is None and browser is not None
        kill_fake(theirs.pid, script)
        theirs.wait(timeout=10)
        assert launcher.get_browser(key) is None
    finally:
        kill_fake(theirs.pid, script)
        launcher.stop_browser(key)


@pytest.mark.live_system_guard_bypass  # a wrongly started second browser is ours to stop
@pytest.mark.parametrize("evidence", ["other-hostname", "other-namespace-record"])
def test_an_unreachable_browser_from_another_environment_is_an_error_not_a_second_launch(tmp_path, evidence):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    udd.mkdir()
    pid = _dead_pid()
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port_file = f"{closed.getsockname()[1]}\n/devtools/browser/elsewhere\n"
    closed.close()  # its DevTools port is the container's loopback: nothing answers here
    (udd / "DevToolsActivePort").write_text(port_file)
    if evidence == "other-hostname":
        _relock(udd, _OTHER_HOST, pid)
    else:
        _relock(udd, socket.gethostname(), pid)  # same hostname (--uts=host), another pid namespace
        _foreign_launch_record(udd, pid)

    try:
        browser, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                ephemeral=False, headed=False, env=dict(os.environ), timeout=30)

        assert browser is None
        assert str(udd) in err and "another environment" in err
        assert pids(udd, "spawns") == [], "a second Chromium was started on a profile another environment holds"
        assert (udd / "DevToolsActivePort").read_text() == port_file
        assert os.readlink(udd / "SingletonLock").endswith(f"-{pid}")
    finally:
        launcher.stop_browser(key)
        for spawned in pids(udd, "spawns"):
            kill_fake(spawned, script)
