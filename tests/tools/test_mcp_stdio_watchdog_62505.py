"""Regression tests for tools/mcp_stdio_watchdog._is_orphaned (#62505 / #62530).

The watchdog must not classify a live Hermes parent as orphaned when
the public psutil ``create_time()`` epoch drifts (WSL2 / system-clock
change, psutil#2526). Only the kernel parent/child relationship
(matching ppid) is used — PPID equality alone is the stable
kernel contract, so the dead create_time / pid_exists chain is gone.
"""

import os
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


def _make_reloaded(monkeypatch, alive=True):
    """Reload the watchdog with a stub psutil; monkeypatch restores."""
    stub = _StubPsutil(alive=alive)
    fake = types.ModuleType("psutil")
    fake.pid_exists = stub.pid_exists
    fake.Process = stub.Process
    fake.Error = _StubPsutil.Error
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    import importlib

    import tools.mcp_stdio_watchdog as wd

    importlib.reload(wd)
    return wd


def test_not_orphaned_when_ppid_matches_and_alive_drifting_ct(monkeypatch):
    # Issue #62505 exact scenario: ppid unchanged + alive, create_time drifted.
    wd = _make_reloaded(monkeypatch, alive=True)
    assert wd._is_orphaned(4242, getppid=lambda: 4242) is False


def test_orphaned_when_ppid_changed(monkeypatch):
    wd = _make_reloaded(monkeypatch, alive=True)
    assert wd._is_orphaned(4242, getppid=lambda: 9999) is True

