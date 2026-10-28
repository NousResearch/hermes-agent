"""The stdio TUI gateway's RPC pipe never reaches a child launched without ``stdin=`` on Windows.

``subprocess`` gives such a child a duplicate of the process's standard input, so the gateway
reads RPC from a private handle and leaves NUL as standard input (``_private_rpc_stdin_windows``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
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
            capture_output=True, text=True, timeout=20)
        seen = child.stdout
    except subprocess.TimeoutExpired:
        seen = "TIMEOUT"
    print("child saw:", seen, flush=True)
    print("gateway read:", repr(sys.stdin.readline()), flush=True)
""")


def test_child_without_stdin_gets_nul_and_the_gateway_keeps_its_requests(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "HERMES_HOME": str(tmp_path / "home")}
    proc = subprocess.Popen([sys.executable, "-c", _GATEWAY], cwd=tmp_path, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdin is not None
    proc.stdin.write('{"id": "req-1"}\n')
    proc.stdin.flush()  # the pipe stays open, as the TUI's does
    try:
        out, err = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    seen = out + err  # importing the gateway points sys.stdout at stderr: stdout is its RPC channel
    assert "child saw: ''" in seen, f"a child without stdin= read the RPC pipe:\n{seen[-3000:]}"
    assert "gateway read: '{\"id\": \"req-1\"}\\n'" in seen, f"the gateway lost its request:\n{seen[-3000:]}"
