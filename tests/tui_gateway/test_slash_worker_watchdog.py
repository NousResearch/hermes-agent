import contextlib
import os
import subprocess
import sys
import time

import pytest

from tui_gateway import slash_worker

def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True

def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_orphan_exit_kill_never_raises(monkeypatch):
    """The kill runs on the watchdog thread right before os._exit: a failure must not stop the exit."""
    from tools.environments import base

    def boom(**_kw):
        raise RuntimeError("kill failed")

    monkeypatch.setattr(base, "kill_live_foreground_processes", boom)
    slash_worker._kill_foreground_commands_before_exit()


def test_orphan_exit_kill_skips_when_terminal_backend_never_loaded(monkeypatch):
    """No command can be live if the backend was never imported; the dying worker must not import it."""
    monkeypatch.delitem(sys.modules, "tools.environments.base", raising=False)
    slash_worker._kill_foreground_commands_before_exit()
    assert "tools.environments.base" not in sys.modules


# Worker stand-in: a foreground terminal command is in flight (it runs in its own session, like every
# local-backend command) when the parent-death watchdog fires and os._exit()s the process.
_ORPHANED_WORKER_CHILD = r"""
import os, sys, threading, time
from tools.environments import base
from tools.environments.local import LocalEnvironment
from tui_gateway import slash_worker
cmd = sys.argv[1]
env = LocalEnvironment(cwd=os.getcwd())
threading.Thread(target=env.execute, args=(cmd,), kwargs={"timeout": 600}, daemon=True).start()
deadline = time.monotonic() + 60
while not base._live_foreground and time.monotonic() < deadline:
    time.sleep(0.02)
if not base._live_foreground:
    sys.exit("foreground command never started")
slash_worker._start_parent_death_watchdog(-1)  # no parent has pid -1: orphaned on the first check
time.sleep(60)
sys.exit("parent-death watchdog never exited the worker")
"""


@pytest.mark.skipif(os.name == "nt", reason="POSIX sessions / reparenting to init")
@pytest.mark.live_system_guard_bypass  # a red run must reap survivors reparented to init
def test_orphaned_worker_takes_its_foreground_command_down(tmp_path):
    """os._exit skips atexit, so the terminal tool's exit cleanup never runs on the watchdog path; a
    foreground command in its own session must still not outlive the orphaned worker."""
    import psutil

    cmd = f"sleep {36000 + os.getpid() % 1000}.7"
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "PYTHONPATH": repo, "HERMES_HOME": str(home),
           "HERMES_SLASH_WATCHDOG_GRACE_S": "0", "HERMES_SLASH_WATCHDOG_POLL_S": "0.05"}
    r = subprocess.run([sys.executable, "-c", _ORPHANED_WORKER_CHILD, cmd], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    survivors = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        survivors = [p for p in psutil.process_iter(["cmdline"])
                     if cmd in " ".join(p.info["cmdline"] or [])]
        if not survivors:
            break
        time.sleep(0.1)
    for p in survivors:
        with contextlib.suppress(psutil.Error):
            p.kill()
    assert not survivors, f"{[' '.join(p.info['cmdline']) for p in survivors]} outlived the orphaned worker"
