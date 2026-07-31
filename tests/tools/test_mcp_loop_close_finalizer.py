"""RED test for Issue B: closing the MCP event loop with a parked server task
still suspended must not emit finalizer noise.

The real trigger: _stop_mcp_loop tore down the loop while a server run task
was parked in _wait_for_reconnect_or_shutdown (never signalled by shutdown,
e.g. a process exit that stops the loop out from under the task). The GC
then finalises the suspended coroutine; its finally block calls t.cancel()
whose call_soon hits a dead loop, printing the unraisable RuntimeError plus
Task-was-destroyed lines.

Runs the reproduction in a subprocess that exercises the real _stop_mcp_loop
production path (start the MCP background loop, park a task on it, then call
_stop_mcp_loop) so the finalizer noise shows up on stderr via sys.unraisablehook.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stop_mcp_loop_drains_parked_tasks_no_finalizer_noise(tmp_path):
    """_stop_mcp_loop must drain parked tasks before closing the loop."""
    f1 = "Event" + " loop is closed"
    f2 = "Task was destroyed" + " but it is pending"
    script = textwrap.dedent("""\
        import asyncio, gc, sys
        from tools.mcp_tool import MCPServerTask, _stop_mcp_loop
        import tools.mcp_tool as mcp_tool

        # Start the real MCP background loop.
        mcp_tool._ensure_mcp_loop()
        loop = mcp_tool._mcp_loop

        async def _park():
            server = MCPServerTask("srv")
            # Park a task on the MCP loop directly.
            park_task = asyncio.ensure_future(
                server._wait_for_reconnect_or_shutdown(timeout=600)
            )
            await asyncio.sleep(0.05)
            assert not park_task.done()
            return park_task, server

        # Schedule the park on the MCP loop and wait for it to suspend.
        import concurrent.futures
        fut = asyncio.run_coroutine_threadsafe(_park(), loop)
        park_task, server = fut.result(timeout=5)

        # Now tear down the loop via the production path WITHOUT signalling
        # shutdown — the process-exit path.
        _stop_mcp_loop()

        del park_task
        del server
        gc.collect()
        gc.collect()
        print("DONE")
    """)
    env = {
        "PYTHONPATH": str(REPO_ROOT),
        "PATH": os.environ.get("PATH", ""),
        "HERMES_HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    stderr = proc.stderr or ""
    stdout = proc.stdout or ""
    noise = [line for line in stderr.splitlines() if f1 in line or f2 in line]
    assert not noise, (
        "MCP loop teardown left finalizer noise on stderr:"
        + "\n".join(noise)
        + f"\n---stdout---\n{stdout}\n---stderr---\n{stderr}"
    )
    assert "DONE" in stdout, f"script did not complete. stdout={stdout!r} stderr={stderr!r}"
