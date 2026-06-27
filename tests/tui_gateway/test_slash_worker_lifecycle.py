"""Tests for slash_worker orphan reaping and lifecycle helpers."""

from __future__ import annotations

import psutil
import pytest

from tui_gateway import slash_worker_lifecycle as lifecycle


def test_is_slash_worker_cmdline():
    assert lifecycle.is_slash_worker_cmdline(
        ["/usr/bin/python", "-m", "tui_gateway.slash_worker", "--session-key", "k"]
    )
    assert not lifecycle.is_slash_worker_cmdline(["python", "-m", "pytest"])


def test_has_live_gateway_owner_true_for_direct_child(monkeypatch):
    my_pid = 4242

    class FakeParent:
        ppid = my_pid

        def cmdline(self):
            return []

    monkeypatch.setattr(lifecycle.psutil, "Process", lambda pid: FakeParent())
    assert lifecycle.has_live_gateway_owner(9001, my_pid=my_pid) is True


def test_has_live_gateway_owner_true_for_gateway_ancestor(monkeypatch):
    gateway_pid = 5000
    worker_pid = 9001

    class GatewayProc:
        ppid = 1

        def cmdline(self):
            return ["python", "-m", "tui_gateway.entry"]

        def status(self):
            return psutil.STATUS_RUNNING

    class WorkerParent:
        ppid = gateway_pid

        def cmdline(self):
            return ["conhost.exe"]

        def status(self):
            return psutil.STATUS_RUNNING

    def fake_process(pid):
        if pid == worker_pid:
            return WorkerParent()
        if pid == gateway_pid:
            return GatewayProc()
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(lifecycle.psutil, "Process", fake_process)
    assert lifecycle.has_live_gateway_owner(worker_pid, my_pid=4242) is True


def test_has_live_gateway_owner_false_when_chain_breaks(monkeypatch):
    worker_pid = 9001

    class WorkerParent:
        ppid = 7777

        def cmdline(self):
            return ["bash"]

        def status(self):
            return psutil.STATUS_RUNNING

    def fake_process(pid):
        if pid == worker_pid:
            return WorkerParent()
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(lifecycle.psutil, "Process", fake_process)
    assert lifecycle.has_live_gateway_owner(worker_pid, my_pid=4242) is False


def test_has_live_gateway_owner_supports_psutil7_ppid_method(monkeypatch):
    my_pid = 4242
    worker_pid = 9001
    gateway_pid = 5000

    class GatewayProc:
        def ppid(self):
            return 1

        def cmdline(self):
            return ["python", "-m", "tui_gateway.entry"]

        def status(self):
            return psutil.STATUS_RUNNING

    class WorkerParent:
        def ppid(self):
            return gateway_pid

        def cmdline(self):
            return ["conhost.exe"]

        def status(self):
            return psutil.STATUS_RUNNING

    def fake_process(pid):
        if pid == worker_pid:
            return WorkerParent()
        if pid == gateway_pid:
            return GatewayProc()
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(lifecycle.psutil, "Process", fake_process)
    assert lifecycle.has_live_gateway_owner(worker_pid, my_pid=my_pid) is True


def test_reap_orphan_slash_workers_terminates_unowned(monkeypatch):
    terminated: list[int] = []

    class FakeProcInfo:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline}

    rows = [
        FakeProcInfo(100, ["python", "-m", "tui_gateway.slash_worker", "--session-key", "k1"]),
        FakeProcInfo(101, ["python", "-m", "pytest"]),
    ]

    monkeypatch.setattr(
        lifecycle.psutil,
        "process_iter",
        lambda attrs: iter(rows),
    )
    monkeypatch.setattr(
        lifecycle,
        "has_live_gateway_owner",
        lambda pid, my_pid: False,
    )
    monkeypatch.setattr(
        lifecycle,
        "_terminate_pid",
        lambda pid: terminated.append(pid),
    )

    count = lifecycle.reap_orphan_slash_workers(my_pid=4242)
    assert count == 1
    assert terminated == [100]


def test_reap_orphan_slash_workers_skips_owned_workers(monkeypatch):
    terminated: list[int] = []

    class FakeProcInfo:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline}

    rows = [
        FakeProcInfo(100, ["python", "-m", "tui_gateway.slash_worker", "--session-key", "k1"]),
    ]

    monkeypatch.setattr(
        lifecycle.psutil,
        "process_iter",
        lambda attrs: iter(rows),
    )
    monkeypatch.setattr(
        lifecycle,
        "has_live_gateway_owner",
        lambda pid, my_pid: True,
    )
    monkeypatch.setattr(
        lifecycle,
        "_terminate_pid",
        lambda pid: terminated.append(pid),
    )

    count = lifecycle.reap_orphan_slash_workers(my_pid=4242)
    assert count == 0
    assert terminated == []


def test_maybe_reap_orphan_slash_workers_on_startup_runs_once(monkeypatch):
    lifecycle._reaper_ran = False
    calls: list[int] = []

    monkeypatch.setattr(
        lifecycle,
        "reap_orphan_slash_workers",
        lambda **kwargs: calls.append(1) or 2,
    )

    lifecycle.maybe_reap_orphan_slash_workers_on_startup()
    lifecycle.maybe_reap_orphan_slash_workers_on_startup()

    assert calls == [1]
