"""Windows-only end-to-end checks for slash_worker lifecycle helpers."""

from __future__ import annotations

import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows job-object e2e")


def test_kill_on_close_job_terminates_child_when_spawner_exits():
    script = """
import subprocess, sys
from tui_gateway.slash_worker_lifecycle import attach_slash_worker_kill_job

BREAKAWAY = 0x01000000
proc = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    creationflags=BREAKAWAY,
)
job = attach_slash_worker_kill_job(proc)
if job is None:
    raise SystemExit("job attach failed")
# Keep job handle alive until this process exits.
_KEEP_JOB = job
print(proc.pid, flush=True)
"""
    spawner = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = spawner.communicate(timeout=15)
    worker_pid = int(out.strip())
    assert spawner.returncode == 0
    time.sleep(2)
    assert not psutil.pid_exists(worker_pid), "child should die when job handle closes"


def test_reaper_kills_orphan_slash_worker_after_launcher_exits():
    launcher = """
import subprocess, sys, time
BREAKAWAY = 0x01000000
proc = subprocess.Popen(
    [sys.executable, "-m", "tui_gateway.slash_worker", "--session-key", "e2e-reap"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    creationflags=BREAKAWAY,
)
print(proc.pid, flush=True)
time.sleep(0.5)
"""
    launcher_proc = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = launcher_proc.communicate(timeout=20)
    worker_pid = int(out.strip())
    # Give watchdog a moment; if still alive, reaper must kill it.
    time.sleep(1)
    if not psutil.pid_exists(worker_pid):
        pytest.skip("watchdog already reaped worker before reaper ran")

    from tui_gateway.slash_worker_lifecycle import reap_orphan_slash_workers

    reaped = reap_orphan_slash_workers(my_pid=__import__("os").getpid())
    time.sleep(1)
    try:
        assert reaped >= 1
        assert not psutil.pid_exists(worker_pid)
    finally:
        if psutil.pid_exists(worker_pid):
            psutil.Process(worker_pid).kill()
