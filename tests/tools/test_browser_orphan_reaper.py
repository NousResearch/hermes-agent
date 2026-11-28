"""Tests for the browser orphan reaper: Chromium launched by a Hermes process that exited without
cleaning up is found through the launcher's per-profile records and stopped, but only after its
identity is proven (``tools.browser_launcher.reap_orphaned_browsers``, driven by
``browser_session_lifecycle.reap_orphaned_browser_sessions``)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.waiting import wait_until
from tools import browser_launcher as launcher
from tools.browser_launcher_holders import ns_identity
from tools import browser_session as bs
from tools import browser_session_lifecycle as lifecycle

_DEAD_OWNER = 99999
_LIVE_OTHER_OWNER = 88888
_REAL_TERMINATE = launcher._terminate
_IGNORES_SIGTERM = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "print('ready', flush=True); time.sleep(60)")


def _stub_terminate(monkeypatch, terminated) -> None:
    """Signal nothing: record the pid, which then reads as gone, like a browser that obeyed its signals."""
    def terminate(pid, start_time, proc):
        terminated.append(pid)
        alive = launcher._pid_alive
        monkeypatch.setattr(launcher, "_pid_alive", lambda other: other != pid and alive(other))
    monkeypatch.setattr(launcher, "_terminate", terminate)


@pytest.fixture
def records(monkeypatch):
    """The test profile's launch records dir; nothing is ever signalled (see ``_stub_terminate``)."""
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    path = home / "cache" / "browser" / "chromium-records"
    path.mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(launcher, "_records_dir", lambda *_home: path)  # this profile's dir, from any caller
    monkeypatch.setattr(launcher, "_served_homes", set(), raising=False)
    monkeypatch.setattr(launcher, "_browsers", {})
    monkeypatch.setattr(launcher, "_procs", {})
    terminated = []
    _stub_terminate(monkeypatch, terminated)
    return SimpleNamespace(dir=path, terminated=terminated,
                           ephemeral=lambda name: str(home / "cache" / "browser" / "chromium" / name))


def _record(records, *, pid: int, udd: str, owner_pid: int = _DEAD_OWNER, start_time=777,
            ephemeral: bool = True) -> Path:
    path = records.dir / f"rec-{pid}.json"
    path.write_text(json.dumps({"owner_pid": owner_pid, "owner_start_time": None, "pid": pid,
                                "start_time": start_time, "user_data_dir": udd, "home_key": "x",
                                "ephemeral": ephemeral, "ns": ns_identity()}))
    return path


def _alive(*pids):
    return lambda pid: pid in pids


class TestReapOrphanedBrowserSessions:
    """Tests for the orphan reaper function."""

    def test_real_profile_browser_is_reaped_when_owner_is_dead_and_its_copy_kept(self, records, tmp_path,
                                                                                 monkeypatch):
        """#100855: the shared real-profile browser is not ``<prefix>_<hex>`` named, so it must still be
        found. Same owner-liveness rule as every other lane: dead owner => reaped. Its persistent
        profile copy is never deleted by the reap."""
        copy_dir = tmp_path / "browser-profile" / "chrome"
        copy_dir.mkdir(parents=True)
        rec = _record(records, pid=4242, udd=str(copy_dir), ephemeral=False)
        monkeypatch.setattr(launcher, "_pid_alive", _alive(4242))  # browser alive, owning hermes gone
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)
        monkeypatch.setattr("tools.browser_lightpanda.reap_orphaned_lightpanda", lambda: 0)

        assert lifecycle.reap_orphaned_browser_sessions() == 1

        assert records.terminated == [4242]
        assert not rec.exists()
        assert copy_dir.exists()

    def test_unfingerprintable_browser_is_refused(self, records, tmp_path, monkeypatch):
        """No start-time fingerprint -> the kill is refused (fail closed).

        A PID whose identity cannot be pinned could be recycled between the verify and the
        tree-kill, so it must be left alone.
        """
        _record(records, pid=12345, udd=records.ephemeral("p"), start_time=None)
        monkeypatch.setattr(launcher, "_pid_alive", _alive(12345))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: None)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)

        launcher.reap_orphaned_browsers()

        assert records.terminated == []


    def test_corrupt_record_is_cleaned(self, records, monkeypatch):
        """A record that is not a JSON object is removed and nothing is signalled."""
        bad = records.dir / "corrupt.json"
        bad.write_text("not-json")
        listed = records.dir / "list.json"
        listed.write_text("[1, 2]")

        assert launcher.reap_orphaned_browsers() == 0
        assert not bad.exists() and not listed.exists()
        assert records.terminated == []


class TestOwnerPidCrossProcess:
    """Owner-based cross-process safe reaping: concurrent Hermes processes never reap each other's
    browsers, so orphans from crashed processes are reaped without touching live ones."""

    def test_alive_owner_is_not_reaped_even_when_untracked(self, records, tmp_path, monkeypatch):
        """A browser with a live owner is NOT reaped, even though this process does not track it.

        This is the core cross-process safety check: Process B scanning while Process A is using a
        browser must not kill A's browser.
        """
        rec = _record(records, pid=12345, udd=records.ephemeral("p"), owner_pid=_LIVE_OTHER_OWNER)
        monkeypatch.setattr(launcher, "_pid_alive", _alive(12345, _LIVE_OTHER_OWNER))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)

        assert launcher.reap_orphaned_browsers() == 0
        assert records.terminated == []
        assert rec.exists()


class TestReaperIdentityGuard:
    """The #14073 guard: before signalling a live PID the reaper confirms it really is the recorded
    browser (its argv names the recorded ``--user-data-dir``), defeating planted records and
    recycled PIDs that would otherwise become an arbitrary same-user DoS."""

    class _FakeProc:
        def __init__(self, cmdline):
            self._cmdline = cmdline

        def cmdline(self):
            return self._cmdline

    def _run(self, cmdline, udd, *, no_such=False, access_denied=False):
        import psutil

        def _factory(pid):
            if no_such:
                raise psutil.NoSuchProcess(pid)
            if access_denied:
                raise psutil.AccessDenied(pid)
            return self._FakeProc(cmdline)

        with patch("psutil.Process", side_effect=_factory):
            return launcher._cmdline_names_dir(12345, udd)

    def test_real_browser_bound_via_cmdline_is_reapable(self):
        udd = "/home/u/.hermes/cache/browser/chromium/abc"
        assert self._run(["chrome", "--remote-debugging-port=0", f"--user-data-dir={udd}"], udd) is True

    def test_recycled_pid_browser_not_bound_to_our_dir_is_refused(self):
        """A Chromium for a DIFFERENT profile dir must not be reaped (PID reuse / an unrelated browser)."""
        udd = "/home/u/.hermes/cache/browser/chromium/abc"
        assert self._run(["chrome", "--user-data-dir=/home/u/.hermes/cache/browser/chromium/OTHER"], udd) is False

    def test_recycled_pid_carrying_only_the_dir_as_a_bare_token_is_refused(self):
        """The dir anywhere in argv is not a binding (#116884): a recycled PID whose argv merely mentions
        it (a grep, a shell) must not pass; only the exact ``--user-data-dir=<dir>`` token binds."""
        udd = "/home/u/.hermes/cache/browser/chromium/abc"
        assert self._run(["grep", udd, "/var/log/syslog"], udd) is False
        assert self._run(["grep", "abc", "/var/log/syslog"], udd) is False
        # Control: the normalized full path as the flag token still binds.
        assert self._run(["chrome", f"--user-data-dir={udd}/"], udd) is True

    def test_unreadable_identity_fails_closed(self):
        udd = "/home/u/.hermes/cache/browser/chromium/abc"
        assert self._run([], udd, no_such=True) is False
        assert self._run([], udd, access_denied=True) is False


class TestLeakedBrowserOfThisProcess:
    """A record owned by THIS (live) process but no longer tracked by it is a leak: reap it. A browser
    this process still tracks is never reaped, however old."""

    def test_untracked_own_browser_is_reaped(self, records, tmp_path, monkeypatch):
        rec = _record(records, pid=12345, udd=records.ephemeral("p"), owner_pid=os.getpid())
        monkeypatch.setattr(launcher, "_pid_alive", _alive(12345, os.getpid()))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)

        assert launcher.reap_orphaned_browsers() == 1
        assert records.terminated == [12345]
        assert not rec.exists()

    def test_tracked_own_browser_is_spared(self, records, tmp_path, monkeypatch):
        udd = records.ephemeral("p")
        rec = _record(records, pid=12345, udd=udd, owner_pid=os.getpid())
        launcher._browsers["mine"] = launcher.LaunchedBrowser(
            key="mine", pid=12345, start_time=777, cdp_url="ws://127.0.0.1:1/devtools/browser/x",
            http_url="http://127.0.0.1:1", user_data_dir=udd, owned=True, ephemeral=True, home_key="x")
        monkeypatch.setattr(launcher, "_pid_alive", _alive(12345, os.getpid()))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)

        assert launcher.reap_orphaned_browsers() == 0
        assert records.terminated == []
        assert rec.exists()

    def test_identity_guard_still_gates_the_leak_path(self, records, tmp_path, monkeypatch):
        """The leak path must not bypass the argv identity check."""
        _record(records, pid=12345, udd=records.ephemeral("p"), owner_pid=os.getpid())
        monkeypatch.setattr(launcher, "_pid_alive", _alive(12345, os.getpid()))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: False)

        assert launcher.reap_orphaned_browsers() == 0
        assert records.terminated == []


def test_throwaway_profile_of_a_crashed_owner_is_removed_even_when_its_browser_died_too(records, monkeypatch):
    """Throwaway profiles are per process, so no later launch wipes a crashed process's dir: the reap does."""
    udd = Path(records.ephemeral("crashed-abc-99999"))
    (udd / "Default").mkdir(parents=True)
    rec = _record(records, pid=4242, udd=str(udd))
    monkeypatch.setattr(launcher, "_pid_alive", _alive())  # owner and browser both gone

    assert launcher.reap_orphaned_browsers() == 0
    assert records.terminated == []
    assert not udd.exists() and not rec.exists()


def test_browser_attached_after_its_owner_crashed_is_never_reaped_while_in_use(records, tmp_path, monkeypatch):
    """The next process ATTACHES to a crashed owner's persistent-profile Chromium (owned=False). The record
    still names the dead owner; this process takes it over, so no reaper kills the browser it handed out."""
    udd = str(tmp_path / "persistent-profile")
    monkeypatch.setattr(launcher, "_pid_alive", _alive(4242))  # the browser survived, its owner did not
    monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
    monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)
    monkeypatch.setattr(launcher, "running_instance", lambda d: (41234, "/devtools/browser/abc", 4242) if d == udd else None)
    from hermes_constants import hermes_home_key
    crashed = launcher.LaunchedBrowser(key=f"profile:{udd}", pid=4242, start_time=777, cdp_url="", http_url="",
                                       user_data_dir=udd, owned=True, ephemeral=False, home_key=hermes_home_key())
    record = launcher._record_path(crashed)  # where the crashed owner left it
    record.write_text(json.dumps({"owner_pid": _DEAD_OWNER, "owner_start_time": None, "pid": 4242, "start_time": 777,
                                  "user_data_dir": udd, "home_key": crashed.home_key, "ephemeral": False,
                                  "ns": ns_identity()}))

    attached, err = launcher.launch_chromium(f"profile:{udd}", executable="/opt/chrome", user_data_dir=udd,
                                             ephemeral=False)
    assert err is None and attached.owned is False

    assert launcher.reap_orphaned_browsers() == 0
    assert records.terminated == []
    assert json.loads(record.read_text())["owner_pid"] == os.getpid()  # other processes see a live owner


class TestPrivatePerProfileRecords:
    """Launcher state lives in a private per-profile dir: a records dir that is not a real directory of this
    user closed to other writers is never read, and the reaper deletes nothing outside the profile's
    ephemeral root (``<HERMES_HOME>/cache/browser/chromium/``)."""

    @staticmethod
    def _orphan(records, monkeypatch, pid=12345):
        rec = _record(records, pid=pid, udd=records.ephemeral("p"))
        monkeypatch.setattr(launcher, "_pid_alive", _alive(pid))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
        monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)
        return rec

    @pytest.mark.platforms("posix")
    def test_records_dir_writable_by_others_is_not_read(self, records, monkeypatch):
        rec = self._orphan(records, monkeypatch)
        records.dir.chmod(0o777)

        assert launcher.reap_orphaned_browsers() == 0
        assert records.terminated == [] and rec.exists()

    @pytest.mark.platforms("posix")
    def test_symlinked_records_dir_is_not_read(self, records, tmp_path, monkeypatch):
        real = tmp_path / "elsewhere"
        records.dir.rename(real)
        records.dir.symlink_to(real, target_is_directory=True)
        rec = self._orphan(records, monkeypatch)

        assert launcher.reap_orphaned_browsers() == 0
        assert records.terminated == [] and rec.exists()

    @pytest.mark.platforms("posix")
    @pytest.mark.live_system_guard_bypass  # the stand-in process is our own child
    def test_ephemeral_record_outside_the_ephemeral_root_is_neither_signalled_nor_deleted(self, records, tmp_path):
        """A live process carrying the recorded ``--user-data-dir=`` token with a matching start time: still
        refused when the "ephemeral" dir is not one the launcher creates."""
        outside = tmp_path / "not-a-launcher-profile"
        (outside / "Default").mkdir(parents=True)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", f"--user-data-dir={outside}"])
        try:
            start_time = wait_until(lambda: launcher._start_time(proc.pid), what="the stand-in's start time")
            _record(records, pid=proc.pid, udd=str(outside), start_time=start_time)

            assert launcher.reap_orphaned_browsers() == 0
            assert records.terminated == []
            assert (outside / "Default").is_dir()
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()


@pytest.mark.platforms("posix")
def test_records_of_every_served_profile_are_reaped(tmp_path, monkeypatch):
    """One process serves many profiles; the janitor/atexit reap runs outside any profile scope. Each served
    home's records are read by path: the launch profile's and a live session's owning profile's."""
    from hermes_constants import get_hermes_home
    home_a, home_b = Path(get_hermes_home()), tmp_path / "profile-b"
    monkeypatch.setattr(launcher, "_browsers", {})
    monkeypatch.setattr(launcher, "_served_homes", set(), raising=False)
    terminated = []
    _stub_terminate(monkeypatch, terminated)
    monkeypatch.setattr(launcher, "_pid_alive", _alive(1111, 2222))
    monkeypatch.setattr(launcher, "_start_time", lambda pid: 777)
    monkeypatch.setattr(launcher, "_cmdline_names_dir", lambda pid, udd: True)
    monkeypatch.setattr("tools.browser_lightpanda.reap_orphaned_lightpanda", lambda: 0)
    for home, pid in ((home_a, 1111), (home_b, 2222)):
        rec_dir = home / "cache" / "browser" / "chromium-records"
        rec_dir.mkdir(mode=0o700, parents=True)
        udd = home / "cache" / "browser" / "chromium" / f"orphan-{pid}"
        udd.mkdir(parents=True)
        _record(SimpleNamespace(dir=rec_dir), pid=pid, udd=str(udd))
    monkeypatch.setitem(bs._session_owner_homes, "task-on-b", str(home_b))

    assert lifecycle.reap_orphaned_browser_sessions() == 2
    assert sorted(terminated) == [1111, 2222]
    assert not (home_b / "cache" / "browser" / "chromium" / "orphan-2222").exists()


class TestRecordOutlivesAnUnprovenBrowser:
    """A launch record is the only handle a later reaper has on its browser, so it and the throwaway profile
    go only once that browser is gone or proven to be something else."""

    @pytest.mark.platforms("linux", "macos")
    def test_browser_that_survives_its_signals_keeps_its_record_and_profile(self, records, monkeypatch):
        """Real process, real signals: SIGTERM is ignored and ``terminal.daemon_term_grace_seconds: 0`` turns off
        the SIGKILL escalation, so the browser outlives the reap. The next pass finishes the job once it is gone."""
        from tools.process_registry import ProcessRegistry
        monkeypatch.setattr(launcher, "_terminate", _REAL_TERMINATE)
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds", staticmethod(lambda: 0))
        monkeypatch.setattr(ProcessRegistry, "_KILL_SETTLE_SECONDS", 0.2)
        udd = Path(records.ephemeral("survivor"))
        (udd / "Default").mkdir(parents=True)
        proc = subprocess.Popen([sys.executable, "-c", _IGNORES_SIGTERM, f"--user-data-dir={udd}"],
                                stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            start_time = wait_until(lambda: launcher._start_time(proc.pid), what="the stand-in's start time")
            rec = _record(records, pid=proc.pid, udd=str(udd), start_time=start_time)

            assert launcher.reap_orphaned_browsers() == 0
            assert proc.poll() is None
            assert rec.exists() and (udd / "Default").is_dir()

            proc.kill()
            proc.wait()
            assert launcher.reap_orphaned_browsers() == 0
            assert not rec.exists() and not udd.exists()
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()

    @pytest.mark.parametrize("unreadable", ["start_time", "argv"])
    def test_browser_whose_identity_cannot_be_read_keeps_its_record_and_profile(self, records, monkeypatch, unreadable):
        """Alive, but its identity cannot be read right now: never signalled, and not forgotten either."""
        import psutil
        udd = Path(records.ephemeral("unreadable"))
        (udd / "Default").mkdir(parents=True)
        rec = _record(records, pid=4242, udd=str(udd))
        monkeypatch.setattr(launcher, "_pid_alive", _alive(4242))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: None if unreadable == "start_time" else 777)
        monkeypatch.setattr(psutil, "Process", lambda pid: (_ for _ in ()).throw(psutil.AccessDenied(pid)))

        assert launcher.reap_orphaned_browsers() == 0

        assert records.terminated == []
        assert rec.exists() and (udd / "Default").is_dir()

    def test_recycled_pid_drops_the_record_without_a_signal(self, records, monkeypatch):
        """Another start time proves the browser is gone: keeping this record would only leak it."""
        udd = Path(records.ephemeral("recycled"))
        (udd / "Default").mkdir(parents=True)
        rec = _record(records, pid=4242, udd=str(udd))
        monkeypatch.setattr(launcher, "_pid_alive", _alive(4242))
        monkeypatch.setattr(launcher, "_start_time", lambda pid: 888)

        assert launcher.reap_orphaned_browsers() == 0

        assert records.terminated == []
        assert not rec.exists() and not udd.exists()


class TestPeriodicOrphanReap:
    """The reaper must run repeatedly, not only at cleanup-thread startup.

    A startup-only reap can never recover from a leak that appears *after*
    boot — which is exactly what happens in a hermes process that stays up
    for days.
    """

    def test_reaper_runs_on_every_interval_not_just_startup(self):
        cycles_to_run = 21
        reap_calls = []
        remaining = {"n": cycles_to_run}

        def fake_cleanup():
            remaining["n"] -= 1
            if remaining["n"] <= 0:
                bs._cleanup_running = False

        orig_running = bs._cleanup_running
        bs._cleanup_running = True
        try:
            with patch("tools.browser_session_lifecycle.reap_orphaned_browser_sessions",
                       side_effect=lambda: reap_calls.append(1)), \
                 patch("tools.browser_session_lifecycle.cleanup_inactive_sessions",
                       side_effect=fake_cleanup), \
                 patch("tools.browser_session_lifecycle.time.sleep"):
                lifecycle._janitor()
        finally:
            bs._cleanup_running = orig_running

        assert len(reap_calls) > 1, "startup-only reap would give exactly 1"
