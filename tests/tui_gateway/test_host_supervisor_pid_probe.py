"""Regression guards for the compute-host supervisor's PID liveness probe.

``host_supervisor._pid_alive`` is called from read-only paths — most notably
``reconcile_startup_orphan``, which only wants to know whether a previously
registered compute host is still running.

On Windows ``os.kill(pid, 0)`` is NOT a no-op: CPython maps sig=0 to
``GenerateConsoleCtrlEvent(0, pid)`` (bpo-14484), which broadcasts Ctrl+C to
the target's *entire console process group*. A status check that can kill the
process it is inspecting — plus unrelated siblings sharing its console — is
the failure mode these tests pin down.

Pattern per the windows-native-support reference / ``TestOwnerAlivePidProbe``
in ``tests/gateway/test_delivery_ledger.py``: patch
``gateway.status._pid_exists``, never ``os.kill``.

Note: ``time.monotonic`` is patched on the shared ``time`` module object
(``hs.time`` IS the stdlib module), so the clock stub is process-wide while the
test runs. It is restored by ``monkeypatch`` on teardown; the advancing
``itertools.count`` keeps _terminate_pid's wait loop bounded regardless of how
many times it is read.
"""

from __future__ import annotations

import itertools

import pytest

from tui_gateway import host_supervisor as hs


# Monotonic clock that advances 0.5s per read, for driving _terminate_pid's
# graceful-wait loop without sleeping in real time. itertools.count never
# raises StopIteration, so an unexpectedly long loop fails the timeout rather
# than masking itself as an error.
_CLOCK = itertools.count(0.0, 0.5)


class TestPidAliveProbe:
    """_pid_alive must route through gateway.status._pid_exists."""

    def test_nonpositive_pid_is_dead_without_probing(self, monkeypatch):
        from gateway import status

        def boom(pid):
            raise AssertionError("probe must not run for a non-positive pid")

        monkeypatch.setattr(status, "_pid_exists", boom)
        assert hs._pid_alive(0) is False
        assert hs._pid_alive(-1) is False

    def test_alive_when_pid_exists(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        assert hs._pid_alive(4242) is True

    def test_dead_when_pid_gone(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        assert hs._pid_alive(4242) is False

    def test_probe_exception_means_dead(self, monkeypatch):
        from gateway import status

        def boom(pid):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(status, "_pid_exists", boom)
        assert hs._pid_alive(4242) is False


class TestTerminatePid:
    """_terminate_pid must route through gateway.status.terminate_pid.

    Windows has no ``signal.SIGKILL`` (AttributeError at import time), and a
    bare SIGTERM there does not cascade to child processes the way
    ``taskkill /T /F`` does.
    """

    def test_force_kill_uses_terminate_pid_force(self, monkeypatch):
        from gateway import status

        calls = []
        monkeypatch.setattr(
            status, "terminate_pid", lambda pid, force=False: calls.append((pid, force))
        )
        # Process never dies, so the graceful wait exhausts and we escalate.
        monkeypatch.setattr(hs, "_pid_alive", lambda pid: True)
        # Drive the wait loop deterministically: an advancing clock (not a
        # frozen one) is what actually terminates it. Patching the module-level
        # _SHUTDOWN_TIMEOUT_SECS has no effect on _terminate_pid's default arg,
        # which was bound at def-time — so pass timeout explicitly.
        monkeypatch.setattr(hs.time, "monotonic", lambda: next(_CLOCK))
        monkeypatch.setattr(hs.time, "sleep", lambda _secs: None)

        hs.HostSupervisor._terminate_pid(None, 4242, timeout=1.0)

        assert calls == [(4242, False), (4242, True)]

    def test_returns_early_when_process_exits(self, monkeypatch):
        from gateway import status

        calls = []
        monkeypatch.setattr(
            status, "terminate_pid", lambda pid, force=False: calls.append((pid, force))
        )
        monkeypatch.setattr(hs, "_pid_alive", lambda pid: False)

        hs.HostSupervisor._terminate_pid(None, 4242)

        assert calls == [(4242, False)]

    def test_missing_process_does_not_escalate(self, monkeypatch):
        from gateway import status

        def gone(pid, force=False):
            raise ProcessLookupError()

        monkeypatch.setattr(status, "terminate_pid", gone)
        # Would raise if we fell through to the force-kill branch.
        hs.HostSupervisor._terminate_pid(None, 4242)


if __name__ == "__main__":
    pytest.main([__file__])
