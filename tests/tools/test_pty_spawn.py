"""A signal aimed at a PTY child before its exec kills the child, and never reaches the parent's loop.

The child keeps the parent's asyncio wakeup fd until asyncio's own ``after_in_child`` fork hook
clears it; a signal that hits the child in that window was handled by the parent's loop as its own
(a SIGTERM shut a gateway down), and one that hits it later was swallowed by asyncio's no-op handler.
The script below holds the post-fork window open in a fresh interpreter: an after-fork hook,
registered before asyncio's, reports the child's pid and waits for the test to signal it. Real PTY,
real processes.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tools import pty_spawn

_HELD_WINDOW_SCRIPT = r"""
import os, sys
ready_r, ready_w = os.pipe()
go_r, go_w = os.pipe()

def hold_post_fork_window():
    os.write(ready_w, f"{os.getpid()}\n".encode())
    os.read(go_r, 1)

os.register_at_fork(after_in_child=hold_post_fork_window)  # runs before asyncio's own hook
import asyncio, signal
from tools.pty_spawn import spawn_pty

async def main(signum, cwd):
    loop = asyncio.get_running_loop()
    seen = []
    loop.add_signal_handler(signum, seen.append, signum)
    spawned = loop.run_in_executor(None, lambda: spawn_pty(["sleep", "10"], cwd=cwd, env=None))
    child = int((await loop.run_in_executor(None, os.read, ready_r, 32)).split()[0])
    os.kill(child, signum)
    os.write(go_w, b"x")
    proc = await spawned
    await loop.run_in_executor(None, proc.wait)
    for _ in range(20):
        await asyncio.sleep(0)  # let the loop drain its wakeup fd
    print(f"parent_saw={bool(seen)} child_signal={proc.signalstatus}")
    loop.remove_signal_handler(signum)

asyncio.run(main(getattr(signal, sys.argv[1]), sys.argv[2]))
"""


def _run_held_window(signal_name, cwd):
    root = Path(pty_spawn.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    return subprocess.run([sys.executable, "-c", _HELD_WINDOW_SCRIPT, signal_name, str(cwd)],
                          cwd=str(root), env=env, capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGUSR1"])
def test_signal_in_post_fork_window_kills_only_the_child(signal_name, tmp_path):
    pytest.importorskip("ptyprocess")
    out = _run_held_window(signal_name, tmp_path)

    assert out.returncode == 0, out.stderr[-2000:]
    signum = int(getattr(signal, signal_name))
    assert f"parent_saw=False child_signal={signum}" in out.stdout, out.stdout


@pytest.mark.platforms("linux")
def test_spawning_thread_signal_mask_is_restored(tmp_path):
    pytest.importorskip("ptyprocess")
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    proc = pty_spawn.spawn_pty(["true"], cwd=str(tmp_path), env=None)
    proc.wait()

    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
