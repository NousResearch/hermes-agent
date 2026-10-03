"""LIVE Windows E2E for MCP orphan tree reaping (#108084).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml`` lane). A stdio MCP
server launched through a ``cmd.exe`` wrapper (npx.cmd -> node.exe) is reaped by signalling the
wrapper PID. ``os.kill`` is TerminateProcess there, so the wrapper's descendants must be
snapshotted before it dies or the grandchild survives the sweep as an orphan.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.platforms("windows")  # live Windows process-tree E2E


def _spawn_wrapper_tree():
    """Start ``cmd.exe /c python -c sleep`` from a short-lived launcher, so this test holds no
    handle to the wrapper (as after the SDK transport closed): returns the wrapper PID. The tree
    gets DEVNULL stdio so it cannot hold the launcher's capture pipe open."""
    launcher = (
        "import subprocess, sys\n"
        "p = subprocess.Popen(['cmd.exe', '/c', sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(p.pid, flush=True)\n"
    )
    out = subprocess.run([sys.executable, "-c", launcher], capture_output=True, text=True, timeout=30)
    return int(out.stdout.strip())


# The wrapper's launcher has exited, so the tree sits outside this test's process subtree; the
# guard would block the real signal this test exists to deliver (only to processes it spawned).
@pytest.mark.live_system_guard_bypass
def test_signalling_wrapper_reaps_its_grandchild():
    import psutil

    import tools.mcp_tool_lifecycle as lifecycle

    wrapper_pid = _spawn_wrapper_tree()
    grandchildren = []
    deadline = time.time() + 30
    while time.time() < deadline and not grandchildren:
        try:
            grandchildren = psutil.Process(wrapper_pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pytest.fail("cmd.exe wrapper exited before its child appeared")
        time.sleep(0.2)
    assert grandchildren, "wrapper never spawned its python child"
    try:
        lifecycle._signal_mcp_process(wrapper_pid, signal.SIGTERM, "live-tree", None, None)
        _, alive = psutil.wait_procs(grandchildren, timeout=15)
        assert not alive, f"descendants survived the wrapper reap: {[p.pid for p in alive]}"
    finally:
        for proc in grandchildren:
            try:
                proc.kill()
            except psutil.Error:
                pass
