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
import textwrap
import time
from types import SimpleNamespace

import pytest

from tools.environments.local import LocalEnvironment


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
        _hermes_pgid=67890,
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )
    killpg_calls = []

    def fake_getpgid(_pid):
        return 67890

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
    assert killpg_calls[0] == (67890, signal.SIGTERM)
    assert (67890, 0) in killpg_calls
