"""Importing tui_gateway.server must not redirect the importing process's stdout.

The stdio sidecar reserves real stdout for JSON-RPC and sends stray print() to stderr, but the
swap belongs to sidecar boot (entry._serve -> _reserve_stdout_for_rpc), not module import:
gateway, agent and CLI processes import helpers from tui_gateway.server, and for
`hermes chat -q` piped to a file or stream-json, stdout IS the product (#141984).
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_importing_server_leaves_stdout_alone():
    """The reporter's guard: a fresh interpreter imports the module and its stdout survives."""
    probe = (
        f"import sys; sys.path.insert(0, {str(REPO)!r}); "
        f"orig = sys.stdout; import tui_gateway.server; "
        f"sys.exit(0 if sys.stdout is orig else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"import redirected stdout (rc={result.returncode})\nstderr: {result.stderr[-2000:]}")


def test_reserve_stdout_for_rpc_sends_print_to_stderr_and_keeps_the_channel():
    """The sidecar boot swap: stdout becomes stderr, and _real_stdout keeps the RPC channel."""
    from tui_gateway import server
    before_stdout = sys.stdout
    before_real = server._real_stdout
    try:
        server._reserve_stdout_for_rpc()
        assert sys.stdout is sys.stderr
        assert server._real_stdout is before_stdout
        # Idempotent at boot: re-running keeps _real_stdout pointed at the RPC channel,
        # never re-pointing it at stderr (which would kill the JSON protocol).
        server._reserve_stdout_for_rpc()
        assert sys.stdout is sys.stderr
        assert server._real_stdout is before_stdout
    finally:
        sys.stdout = before_stdout
        server._real_stdout = before_real
