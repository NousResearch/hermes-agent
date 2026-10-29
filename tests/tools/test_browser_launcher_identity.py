"""A browser in launcher state is ONE launch: its PID namespace and boot, pid, process start time and profile
(``--user-data-dir`` realpath), never its pid number alone.

Records from earlier containers or boots are kept (their pids cannot be judged from here), so a pid number can
recur across them, and two containers can give two different browsers the same number. An attacher leases the
launch it actually reached (for a browser of another environment: the record whose DevTools browser id its
endpoint serves), and stopping one key lets go only of that key's own browser (``tools/browser_launcher.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = pytest.mark.platforms("linux")

_REPO = Path(__file__).resolve().parents[2]
_OTHER_HOST = "container-3f9c2a"
_OLD_NS = "4026531836:00000000-0000-0000-0000-000000000000"  # a container (or boot) that is gone
_LIVE_NS = "4026532999:11111111-1111-1111-1111-111111111111"  # the container running the browser now


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _start_fake(script, udd) -> subprocess.Popen:
    proc = subprocess.Popen([str(script), "--remote-debugging-port=0", f"--user-data-dir={udd}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert wait_until(lambda: (udd / "DevToolsActivePort").exists(), timeout=15, what="the fake to come up")
    return proc


def _as_container_browser(udd, pid: int) -> None:
    """Make the profile's SingletonLock name ``pid`` under a container's hostname: its number in THAT namespace."""
    lock = udd / "SingletonLock"
    lock.unlink(missing_ok=True)
    os.symlink(f"{_OTHER_HOST}-{pid}", lock)


def _record(udd, pid: int, tag: str, ns: str, start_time: int, devtools) -> Path:
    """The launch record a Hermes in namespace ``ns`` wrote for its browser ``pid`` on ``udd``."""
    path = launcher._shared_state_dir(str(udd)) / f"{launcher._profile_digest(str(udd))[:16]}-{pid}{tag}.json"
    data = {"owner_pid": pid + 1, "owner_start_time": 1, "pid": pid, "start_time": start_time,
            "user_data_dir": str(udd), "home_key": "elsewhere", "ephemeral": False, "ns": ns}
    path.write_text(json.dumps({**data, **({"devtools": devtools} if devtools else {})}))
    return path


# The owner's own view: ``other_live_holders`` run as a process of the container that launched the browser.
_OWNER_VIEW = textwrap.dedent("""\
    import sys
    from pathlib import Path
    from tools import browser_launcher_holders as holders
    holders.ns_identity = lambda: sys.argv[2]
    print(len(holders.other_live_holders(Path(sys.argv[1]), int(sys.argv[3]))), flush=True)
""")


def _owner_sees(record: Path, ns: str, start_time: int) -> int:
    out = subprocess.run([sys.executable, "-c", _OWNER_VIEW, str(record), ns, str(start_time)], cwd=_REPO,
                         env=dict(os.environ), capture_output=True, text=True, timeout=60, check=True)
    return int(out.stdout.strip())


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
@pytest.mark.parametrize("old_devtools", ["/devtools/browser/an-earlier-launch", None],
                         ids=["old-launch-was-ready", "old-launch-never-ready"])
def test_an_older_record_on_the_same_pid_does_not_take_the_live_browsers_holders(tmp_path, old_devtools):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    theirs = _start_fake(script, udd)
    try:
        pid = _dead_pid()  # the container's number for its browser: names nobody here
        _as_container_browser(udd, pid)
        live_path = (udd / "DevToolsActivePort").read_text().split()[1]
        # Sorted first: a record an earlier container (or boot) left for ITS browser on the same pid number.
        old = _record(udd, pid, "-n000000000000", _OLD_NS, 111, old_devtools)
        live = _record(udd, pid, "-nffffffffffff", _LIVE_NS, 222, live_path)
        old_bytes = old.read_bytes()

        browser, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                ephemeral=False, headed=False, env=dict(os.environ), timeout=30)

        assert err is None and browser is not None and browser.foreign
        assert _owner_sees(live, _LIVE_NS, 222) == 1, "the live browser's owner does not see the attacher"
        assert not list(old.parent.glob(f"{old.stem}@*.lease")), "the attacher leased an earlier launch"
        assert old.read_bytes() == old_bytes, "the earlier launch's record was rewritten"

        launcher.stop_browser(key)
        assert not list(live.parent.glob("*.lease"))
        assert theirs.poll() is None, "stopped a browser another environment holds"
    finally:
        launcher.stop_browser(key)
        kill_fake(theirs.pid, script)
        theirs.wait(timeout=10)


@pytest.mark.live_system_guard_bypass  # the fake browsers are ours to stop if a leg failed
def test_stopping_one_of_two_same_pid_browsers_drops_only_its_own_lease(tmp_path):
    """Two containers' browsers on two pinned profiles, each pid N in its own namespace: one session ending must
    let go of its browser only, not leave its lease behind because the other key looked like the same browser."""
    script = write_fake_chromium(tmp_path / "chromium")
    pid = _dead_pid()
    profiles = [tmp_path / "profile-a", tmp_path / "profile-b"]
    fakes = [_start_fake(script, udd) for udd in profiles]
    keys = [f"profile:{udd}" for udd in profiles]
    try:
        for udd in profiles:
            _as_container_browser(udd, pid)
        for key, udd in zip(keys, profiles):
            browser, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                    ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
            assert err is None and browser is not None and browser.foreign and browser.pid == pid
        states = [launcher._shared_state_dir(str(udd)) for udd in profiles]
        assert all(list(state.glob("*.lease")) for state in states)

        launcher.stop_browser(keys[0])

        assert not list(states[0].glob("*.lease")), "the stopped session left its lease: its browser is held forever"
        assert list(states[1].glob("*.lease")), "stopping one browser dropped the other one's lease"
        assert launcher.get_browser(keys[1]) is not None
        assert all(fake.poll() is None for fake in fakes)
    finally:
        for key in keys:
            launcher.stop_browser(key)
        for fake in fakes:
            kill_fake(fake.pid, script)
            fake.wait(timeout=10)


@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_lease_on_an_earlier_launch_does_not_keep_the_current_browser_up(tmp_path):
    """A holder in another namespace leased an earlier browser filed under the same record name (that namespace's
    pid recycled): its lease names that launch's start time, not this one's, so the last real holder still stops
    the browser it launched."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd, key = tmp_path / "pinned-profile", f"profile:{tmp_path / 'pinned-profile'}"
    mine, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd), ephemeral=False,
                                         headed=False, env=dict(os.environ), timeout=30)
    assert err is None and mine.owned
    try:
        record, gone_here = launcher._record_path(mine), _dead_pid()
        stale = record.with_name(f"{record.stem}@{gone_here}-nffffffffffff.lease")
        stale.write_text(json.dumps({"holder_pid": gone_here, "holder_start_time": None, "pid": mine.pid,
                                     "start_time": mine.start_time - 1, "user_data_dir": str(udd), "ns": _LIVE_NS}))

        launcher.stop_browser(key)

        assert wait_until(lambda: not launcher._pid_alive(mine.pid), timeout=15, what="the browser to stop")
        assert not record.exists() and not stale.exists()
    finally:
        kill_fake(mine.pid, script)
