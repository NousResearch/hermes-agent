"""Regression tests for #85125 Phase 4b (terminal flavor of the #71148 class).

LocalEnvironment._kill_process kills the process GROUP (SIGTERM -> wait ->
SIGKILL).  A descendant that called ``setsid`` escapes the group and survives
the group-kill — the local sibling of issue #84967.  The fix snapshots the
descendant set via psutil BEFORE the first signal (children reparent to init
after the parent dies, so a later parent walk finds nothing — same rationale
as agent/deadline.py kill_process_tree) and sweeps any snapshotted survivor
outside the (now-dead) group with SIGKILL afterwards.
"""

import os
import signal
import subprocess
import textwrap
import time
from types import SimpleNamespace

import pytest

from tools.environments.local import LocalEnvironment, _kill_process_group_posix


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_pid_exit(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


@pytest.mark.live_system_guard_bypass
def test_timeout_kill_reaps_setsid_grandchild(tmp_path):
    """A grandchild that setsid's out of the group must not survive the
    timeout kill path."""
    pytest.importorskip("psutil")

    pid_file = tmp_path / "grandchild.pid"
    script = textwrap.dedent(
        """
        import os, sys, time
        pid = os.fork()
        if pid == 0:
            os.setsid()  # escape the command's process group/session
            with open(sys.argv[1], "w") as f:
                f.write(str(os.getpid()))
            time.sleep(30)
            os._exit(0)
        time.sleep(30)
        """
    ).strip()

    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        import sys as _sys

        cmd = f"{_sys.executable} -c {_sh_quote(script)} {_sh_quote(str(pid_file))}"
        result = env.execute(cmd, timeout=3)

        # The command must have hit the timeout/kill path.
        assert "timed out" in result.get("output", "").lower() or result.get(
            "returncode"
        ) not in (0,), f"expected timeout, got: {result!r}"

        # The grandchild wrote its pid before the kill.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        assert pid_file.exists(), "grandchild never wrote its pid file"
        grandchild_pid = int(pid_file.read_text().strip())

        assert _wait_for_pid_exit(grandchild_pid), (
            f"setsid grandchild {grandchild_pid} SURVIVED the timeout "
            f"group-kill — the #84967/#71148 orphan class (terminal flavor). "
            f"_kill_process must sweep snapshotted descendants outside the "
            f"group after the group-kill."
        )
    finally:
        # Belt and braces: never leak the sleeper into the test host.
        try:
            if pid_file.exists():
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
        except (OSError, ValueError):
            pass
        try:
            env.cleanup()
        except Exception:
            pass


def _sh_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def test_kill_process_survives_psutil_snapshot_failure(monkeypatch):
    """A broken psutil snapshot must never break the kill path — the
    group-kill escalation still runs to completion."""
    psutil = pytest.importorskip("psutil")

    env = object.__new__(LocalEnvironment)
    proc = SimpleNamespace(
        pid=12345,
        _hermes_pgid=12345,  # wrapper leads its own group (start_new_session)
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )
    killpg_calls = []

    def fake_getpgid(_pid):
        return 12345

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError  # group is gone after the first signal

    def boom(*_a, **_k):
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    monkeypatch.setattr(psutil, "Process", boom)

    env._kill_process(proc)  # must not raise

    # SIGTERM was delivered to the group and the alive-probe ran: the
    # escalation path completed despite the snapshot failure.
    assert killpg_calls[0] == (12345, signal.SIGTERM)
    assert (12345, 0) in killpg_calls


def test_kill_process_swallows_killpg_permissionerror(monkeypatch):
    """#116855: on macOS a group that empties between the caller's liveness
    check and the TERM raises PermissionError (not ESRCH) from ``killpg``.
    The teardown must skip the escalation instead of raising, exactly like
    the vanished-group case."""
    psutil = pytest.importorskip("psutil")

    proc = SimpleNamespace(
        pid=12345,
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )
    killpg_calls = []

    def fake_getpgid(_pid):
        return 12345  # rg spawns with start_new_session: it LEADS the group

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    monkeypatch.setattr(psutil, "Process", lambda _pid: (_ for _ in ()).throw(RuntimeError("no psutil")))

    # Direct call, matching the rg teardown in file_operations_search
    # (_run_rg_native): that path has no _kill_process OSError wrapper.
    _kill_process_group_posix(proc)  # must not raise

    # Only the TERM was attempted; the EPERM (zombie-only group: wrapper
    # exited, nothing to signal) skip means no probe, no SIGKILL.
    assert killpg_calls == [(12345, signal.SIGTERM)]


def test_child_not_leading_its_group_never_killpgs(monkeypatch):
    """#107029: a wrapper spawned without setsid shares the CALLER's process
    group — on Darwin, the gateway's. Killing "its" group would signal the
    caller itself, so teardown must never ``killpg`` and must take the child
    down by PID."""
    pytest.importorskip("psutil")
    proc = subprocess.Popen(["sleep", "30"])  # no start_new_session: our group

    killpg_calls = []

    def recording_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))  # the real call would kill OUR group

    monkeypatch.setattr(os, "killpg", recording_killpg)
    try:
        _kill_process_group_posix(proc)  # getpgid(proc.pid) == our pgid != pid
        assert killpg_calls == []
        assert proc.poll() is not None  # taken down by PID (terminate)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_genuine_killpg_eperm_falls_back_to_pid_teardown(monkeypatch):
    """#104696: EPERM alone must not be read as "group gone". While the child
    itself still lives, the failed group TERM must not skip the escalation
    (the same-pgid sweep skip would leak it) — teardown falls back to PID
    kills."""
    pytest.importorskip("psutil")
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)

    def eperm_killpg(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", eperm_killpg)
    try:
        _kill_process_group_posix(proc)  # TERM EPERMs against a live group
        assert proc.poll() is not None  # PID teardown killed the live child
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_unreaped_wrapper_without_cached_pgid_falls_back_to_pid(monkeypatch):
    """A wrapper whose getpgid() ESRCHs with no spawn-cached pgid used to
    raise out of the teardown; it must fall back to PID teardown instead —
    the _kill_process OSError wrapper is not on every caller (#116855's rg
    path calls the helper directly)."""
    pytest.importorskip("psutil")
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)

    killpg_calls = []

    def esrch_getpgid(_pid):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "getpgid", esrch_getpgid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    try:
        _kill_process_group_posix(proc)  # pgid None != pid -> PID teardown
        assert killpg_calls == []
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_group_members_gone_truth_table():
    """The EPERM disambiguation probe (#116855 vs #104696): only a fully
    exited (or zombie-only) member set proves the group emptied."""
    psutil = pytest.importorskip("psutil")
    from tools.environments.local import _group_members_gone

    live_proc = SimpleNamespace(poll=lambda: None)
    dead_proc = SimpleNamespace(poll=lambda: 0)
    zombie_child = SimpleNamespace(status=lambda: psutil.STATUS_ZOMBIE)
    running_child = SimpleNamespace(status=lambda: psutil.STATUS_RUNNING)

    def _raise_gone():
        raise psutil.NoSuchProcess(1)

    gone_child = SimpleNamespace(status=_raise_gone)

    assert _group_members_gone(live_proc, []) is False  # child itself lives
    assert _group_members_gone(dead_proc, []) is True
    assert _group_members_gone(dead_proc, [zombie_child]) is True
    assert _group_members_gone(dead_proc, [zombie_child, gone_child]) is True
    assert _group_members_gone(dead_proc, [running_child]) is False  # live member
