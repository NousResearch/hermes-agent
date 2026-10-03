"""Killing a PTY background process must return, and must reach descendants that escape it.

An escapee (a descendant that ``setsid()``s into its own session) keeps the PTY slave open, so
the reader thread stays blocked in ``read()`` holding the PTY file object's buffer lock. Closing
the PTY from the kill path waited on that lock until the escapee exited (forever, for a
long-lived one), so the kill never returned. And unless the process runs in its own systemd
scope, nothing reaps the escapee: a supervised dashboard / serve backend now gets that scope
like the supervised gateway does. Real PTY, real processes: a fake PTY cannot hold the lock.
"""

import os
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import tools.process_registry as module
from tools.process_registry import ProcessRegistry

_POSIX_PTY = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("setsid") is None, reason="POSIX PTY + setsid")

# The unscoped escapee exits on its own (it is reparented to init, outside what a test may
# signal). Before the fix the kill could only return once it had, so the kill deadline is shorter.
_ESCAPEE_LIFETIME_S = 8
_KILL_DEADLINE_S = 5


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _spawn_with_escapee(registry, tmp_path, escapee_lifetime):
    pidfile = tmp_path / "escapee.pid"
    session = registry.spawn_local(
        f"setsid sh -c 'echo $$ > {pidfile}; exec sleep {escapee_lifetime}' & sleep 120",
        cwd=str(tmp_path), use_pty=True)
    assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), 5)
    return session, int(pidfile.read_text())


def _kill_within_deadline(registry, session):
    result = {}
    killer = threading.Thread(
        target=lambda: result.update(registry.kill_process(session.id)), daemon=True)
    killer.start()
    killer.join(_KILL_DEADLINE_S)
    assert not killer.is_alive(), "kill_process blocked closing the PTY under a live reader"
    return result


@_POSIX_PTY
def test_kill_returns_while_escaped_descendant_holds_the_pty(tmp_path, monkeypatch):
    pytest.importorskip("ptyprocess")
    # No systemd scope: stopping one would reap the escapee and hide the hang.
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(module, "_is_supervised_backend_host", lambda: False, raising=False)
    registry = ProcessRegistry()
    session, _ = _spawn_with_escapee(registry, tmp_path, _ESCAPEE_LIFETIME_S)

    result = _kill_within_deadline(registry, session)
    assert result["status"] == "killed"
    assert session.id in registry._finished

    # Once the escapee exits, the reader's read ends and it closes the PTY itself, so the
    # master FD is still released.
    reader = session._reader_thread
    assert reader is not None
    assert _wait_for(lambda: not reader.is_alive(), _ESCAPEE_LIFETIME_S + 5)
    assert session._pty.closed


@pytest.mark.parametrize("marked, supervised, scoped", [
    (True, True, True),     # dashboard / serve run by a service manager
    (True, False, False),   # dashboard / serve started from a shell
    (False, True, False),   # a CLI or terminal child that only inherited the supervisor env
])
def test_supervised_backend_host_scopes_background_processes(monkeypatch, marked, supervised, scoped):
    monkeypatch.setattr(module, "_IS_LINUX", True)
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(module, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr(module, "_build_systemd_scope_argv",
                        lambda argv, unit_suffix: ["systemd-run", "--unit", unit_suffix, *argv])
    monkeypatch.setattr(module, "_backend_host", marked, raising=False)
    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", lambda *a, **k: supervised)
    session = SimpleNamespace(systemd_unit="")

    argv = ProcessRegistry._scope_argv(ProcessRegistry(), session, "echo hi", "proc_x", "PTY")

    assert (argv[0] == "systemd-run") is scoped
    assert bool(session.systemd_unit) is scoped


@_POSIX_PTY
def test_scoped_backend_kill_reaps_an_escaped_descendant(tmp_path, monkeypatch):
    pytest.importorskip("ptyprocess")
    if sys.platform != "linux" or not module._systemd_run_user_scope_available():
        pytest.skip("needs systemd-run --user --scope (a reachable user bus)")
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(module, "_backend_host", True, raising=False)
    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", lambda *a, **k: True)
    registry = ProcessRegistry()
    session, escapee = _spawn_with_escapee(registry, tmp_path, 120)
    try:
        assert session.systemd_unit.startswith("hermes-worker-")
        result = _kill_within_deadline(registry, session)
        assert result["status"] == "killed"
        # Stopping the scope reaps the whole cgroup, the escapee included.
        assert _wait_for(lambda: not _pid_alive(escapee), 10), "escaped descendant survived the kill"
    finally:
        if session.systemd_unit:
            module._stop_systemd_unit(session.systemd_unit)
