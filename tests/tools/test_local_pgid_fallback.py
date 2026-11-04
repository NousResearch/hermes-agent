"""OpenBSD getpgid(2) returns EPERM across sessions. The spawn must still record a start time."""

from __future__ import annotations

import os

from tools.environments.local import LocalEnvironment


def test_spawn_does_not_invent_pgid_when_process_is_gone(monkeypatch):
    env = LocalEnvironment(cwd="/tmp", timeout=15, persistent=False)

    def gone(pid):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", gone)
    proc = env._run_bash("true", timeout=15)
    proc.wait(timeout=15)
    assert getattr(proc, "_hermes_pgid", None) is None
    assert getattr(proc, "_hermes_pgid_start", None) is None


def test_spawn_records_pgid_start_when_getpgid_succeeds(monkeypatch):
    env = LocalEnvironment(cwd="/tmp", timeout=15, persistent=False)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    proc = env._run_bash("true", timeout=15)
    proc.wait(timeout=15)
    assert proc._hermes_pgid == proc.pid
    assert getattr(proc, "_hermes_pgid_start", None) is not None
