"""The stdio TUI gateway's RPC pipe never reaches a child launched without ``stdin=`` on Windows.

``subprocess`` gives such a child a duplicate of the process's standard input, so the gateway
reads RPC from a private handle and leaves NUL as standard input (``_private_rpc_stdin_windows``).
The TUI keeps the pipe open while the gateway works, so a child holding that duplicate blocks
on it instead of seeing EOF: the test keeps its writer open the same way.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[2]

_GATEWAY = textwrap.dedent("""
    import subprocess, sys
    from tui_gateway.entry import _close_rpc_stdin_on_exec
    _close_rpc_stdin_on_exec()
    try:
        child = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stdout.write(repr(sys.stdin.read()))"],
            capture_output=True, text=True, timeout=10)
        seen = child.stdout
    except subprocess.TimeoutExpired:
        seen = "TIMEOUT"
    print("child saw:", seen, flush=True)
    print("gateway read:", repr(sys.stdin.readline()), flush=True)
""")


def _next_line(lines: queue.Queue, prefix: str, transcript: list[str], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        try:
            line = lines.get(timeout=left)
        except queue.Empty:
            break
        if line is None:
            break
        transcript.append(line)
        if line.startswith(prefix):
            return line.rstrip("\n")
    pytest.fail(f"no {prefix!r} line from the gateway:\n{''.join(transcript)[-3000:]}")


def test_child_without_stdin_sees_nul_while_the_rpc_pipe_stays_open(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "HERMES_HOME": str(tmp_path / "home")}
    # Importing the gateway points sys.stdout at stderr (stdout is its RPC channel): merge them.
    proc = subprocess.Popen([sys.executable, "-c", _GATEWAY], cwd=tmp_path, env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdin is not None and proc.stdout is not None
    out = proc.stdout
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in out:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    transcript: list[str] = []
    try:
        # No request written yet and the writer open: a child holding the RPC pipe blocks here.
        saw = _next_line(lines, "child saw:", transcript, timeout=60)
        assert saw == "child saw: ''", f"a child without stdin= held the RPC pipe:\n{''.join(transcript)}"

        proc.stdin.write('{"id": "req-1"}\n')
        proc.stdin.flush()
        read = _next_line(lines, "gateway read:", transcript, timeout=30)
        assert read == "gateway read: '{\"id\": \"req-1\"}\\n'", f"the gateway lost its request:\n{''.join(transcript)}"

        proc.stdin.close()
        assert proc.wait(timeout=30) == 0, "".join(transcript)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
