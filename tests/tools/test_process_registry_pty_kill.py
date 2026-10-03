"""Killing a PTY process must return even when a descendant escaped into its own session.

The escapee keeps the PTY slave open, so the reader thread stays blocked in ``read()`` holding
the PTY file object's buffer lock. Closing the PTY from the kill path waited on that lock until
the escapee exited (forever, for a long-lived one), so the kill never returned. Real PTY, real
processes: a fake PTY cannot hold the lock.
"""

import shutil
import sys
import threading
import time

import pytest

import tools.process_registry as module
from tools.process_registry import ProcessRegistry

pytest.importorskip("ptyprocess")
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("setsid") is None, reason="POSIX PTY + setsid")

# The escapee exits on its own (it is reparented to init, outside what a test may signal).
# Before the fix the kill could only return once it had, so the kill deadline is shorter.
_ESCAPEE_LIFETIME_S = 8
_KILL_DEADLINE_S = 5


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_kill_returns_while_escaped_descendant_holds_the_pty(tmp_path, monkeypatch):
    # No systemd scope: stopping one would reap the escapee and hide the hang.
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)
    pidfile = tmp_path / "escapee.pid"
    registry = ProcessRegistry()
    session = registry.spawn_local(
        f"setsid sh -c 'echo $$ > {pidfile}; exec sleep {_ESCAPEE_LIFETIME_S}' & sleep 120",
        cwd=str(tmp_path), use_pty=True)
    assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), 5)

    result = {}
    killer = threading.Thread(
        target=lambda: result.update(registry.kill_process(session.id)), daemon=True)
    killer.start()
    killer.join(_KILL_DEADLINE_S)
    assert not killer.is_alive(), "kill_process blocked closing the PTY under a live reader"
    assert result["status"] == "killed"
    assert session.id in registry._finished

    # Once the escapee exits, the reader's read ends and it closes the PTY itself, so the
    # master FD is still released.
    reader = session._reader_thread
    assert reader is not None
    assert _wait_for(lambda: not reader.is_alive(), _ESCAPEE_LIFETIME_S + 5)
    assert session._pty.closed
