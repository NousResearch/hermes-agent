"""Regression tests for tools/mcp_stdio_watchdog._is_orphaned (#62505).

The watchdog must not classify a live Hermes parent as orphaned when the
public psutil ``create_time()`` epoch drifts (WSL2 / system-clock change,
psutil#2526). Only the kernel parent/child relationship (matching ppid +
pid still exists) is used.
"""
import sys
import types

import pytest


class _StubPsutil:
    """Minimal psutil stand-in: pid_exists reflects a live parent."""

    Error = Exception

    def __init__(self, alive=True):
        self._alive = alive

    def pid_exists(self, pid):
        return self._alive

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 101.0


@pytest.fixture()
def watchdog_with_stub(monkeypatch):
    """Import the module with a stub psutil so the psutil branch runs."""
    stub = _StubPsutil(alive=True)
    # Build a fake module so `import psutil` inside the target resolves to stub.
    fake = types.ModuleType("psutil")
    fake.pid_exists = stub.pid_exists
    fake.Process = stub.Process
    fake.Error = _StubPsutil.Error
    monkeypatch.setitem(sys.modules, "psutil", fake)
    import importlib

    import tools.mcp_stdio_watchdog as wd

    importlib.reload(wd)
    yield wd
    monkeypatch.setitem(sys.modules, "psutil", None)
    importlib.reload(wd)


def test_not_orphaned_when_ppid_matches_and_alive_drifting_ct(watchdog_with_stub):
    # Issue #62505 exact scenario: ppid unchanged + alive, create_time drifted.
    assert watchdog_with_stub._is_orphaned(4242, 100.0, getppid=lambda: 4242) is False


def test_orphaned_when_ppid_changed(watchdog_with_stub):
    assert watchdog_with_stub._is_orphaned(4242, 100.0, getppid=lambda: 9999) is True


def test_orphaned_when_parent_pid_gone(watchdog_with_stub, monkeypatch):
    stub = _StubPsutil(alive=False)
    fake = types.ModuleType("psutil")
    fake.pid_exists = stub.pid_exists
    fake.Process = stub.Process
    fake.Error = _StubPsutil.Error
    monkeypatch.setitem(sys.modules, "psutil", fake)
    import importlib

    import tools.mcp_stdio_watchdog as wd

    importlib.reload(wd)
    assert wd._is_orphaned(4242, 100.0, getppid=lambda: 4242) is True
