"""Windows background process-tree cleanup regression tests.

These tests use fake psutil process objects only. They must never kill a real
node/npm/cmd process by executable name alone.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import psutil

from tools import process_registry as pr
from tools.process_registry import ProcessRegistry


class FakeWindowsProcess:
    def __init__(
        self,
        pid: int,
        *,
        cmdline: list[str],
        cwd: str,
        create_time: float,
        name: str | None = None,
        alive: bool = True,
        stubborn: bool = False,
    ):
        self.pid = pid
        self._cmdline = cmdline
        self._cwd = cwd
        self._create_time = create_time
        self._name = name or (Path(cmdline[0]).name if cmdline else "process.exe")
        self.alive = alive
        self.stubborn = stubborn
        self._children: list[FakeWindowsProcess] = []
        self._parent: FakeWindowsProcess | None = None
        self.taskkill_count = 0

    def add_child(self, child: "FakeWindowsProcess") -> "FakeWindowsProcess":
        child._parent = self
        self._children.append(child)
        return child

    def children(self, recursive: bool = False):
        if not recursive:
            return list(self._children)
        out = []
        stack = list(self._children)
        while stack:
            child = stack.pop(0)
            out.append(child)
            stack[0:0] = child._children
        return out

    def parent(self):
        return self._parent

    def parents(self):
        out = []
        cur = self._parent
        while cur is not None:
            out.append(cur)
            cur = cur._parent
        return out

    def cmdline(self):
        return list(self._cmdline)

    def cwd(self):
        return self._cwd

    def create_time(self):
        return self._create_time

    def name(self):
        return self._name

    def is_running(self):
        return self.alive

    def status(self):
        return "running" if self.alive else "terminated"

    def mark_taskkill(self):
        self.taskkill_count += 1
        if not self.stubborn:
            self.alive = False

    def terminate(self):
        self.mark_taskkill()

    def kill(self):
        self.mark_taskkill()

    def net_connections(self, kind="inet"):
        return getattr(self, "_connections", [])

    def connections(self, kind="inet"):
        return self.net_connections(kind)


def _fake_listener(port: int):
    return types.SimpleNamespace(
        status="LISTEN",
        laddr=types.SimpleNamespace(port=port),
    )


def _make_dev_tree(project_dir: str):
    root = FakeWindowsProcess(
        1000,
        cmdline=["C:/Program Files/Git/bin/bash.exe", "-lic", "set +m; npm run dev"],
        cwd=project_dir,
        create_time=100.0,
        name="bash.exe",
    )
    cmd = root.add_child(FakeWindowsProcess(
        1001,
        cmdline=["C:/Windows/System32/cmd.exe", "/d", "/s", "/c", "npm run dev"],
        cwd=project_dir,
        create_time=101.0,
        name="cmd.exe",
    ))
    npm = cmd.add_child(FakeWindowsProcess(
        1002,
        cmdline=["C:/Program Files/nodejs/node.exe", "C:/Program Files/nodejs/npm-cli.js", "run", "dev"],
        cwd=project_dir,
        create_time=102.0,
        name="node.exe",
    ))
    next_dev = npm.add_child(FakeWindowsProcess(
        1003,
        cmdline=["C:/Program Files/nodejs/node.exe", "node_modules/next/dist/bin/next", "dev"],
        cwd=project_dir,
        create_time=103.0,
        name="node.exe",
    ))
    start_server = next_dev.add_child(FakeWindowsProcess(
        1004,
        cmdline=["C:/Program Files/nodejs/node.exe", "node_modules/next/dist/server/lib/start-server.js"],
        cwd=project_dir,
        create_time=104.0,
        name="node.exe",
    ))
    return root, cmd, npm, next_dev, start_server


def test_windows_tree_kill_targets_owned_local_dev_descendants_not_generic_node(monkeypatch, tmp_path):
    project_dir = str(tmp_path / "app")
    other_dir = str(tmp_path / "other")
    root, cmd, npm, next_dev, start_server = _make_dev_tree(project_dir)
    generic_node = FakeWindowsProcess(
        2000,
        cmdline=["C:/Program Files/nodejs/node.exe", "unrelated-worker.js"],
        cwd=other_dir,
        create_time=105.0,
        name="node.exe",
    )

    by_pid = {p.pid: p for p in [root, cmd, npm, next_dev, start_server, generic_node]}
    taskkill_pids: list[int] = []

    def fake_process(pid):
        return by_pid[pid]

    def fake_run(args, **kwargs):
        pid = int(args[args.index("/PID") + 1])
        taskkill_pids.append(pid)
        by_pid[pid].mark_taskkill()
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pr, "_IS_WINDOWS", True)
    monkeypatch.setattr(ProcessRegistry, "_host_pid_is_ours", classmethod(lambda cls, pid, expected_start: pid == 1000))
    monkeypatch.setattr(pr.psutil, "Process", fake_process)
    monkeypatch.setattr(pr.subprocess, "run", fake_run)

    result = ProcessRegistry._terminate_host_pid(1000, expected_start=123456)

    assert result["status"] == "killed"
    assert set(taskkill_pids) >= {1000, 1001, 1002, 1003, 1004}
    assert 2000 not in taskkill_pids, "generic node outside the owned tree must not be killed by name"
    assert generic_node.alive is True


def test_windows_tree_kill_reports_partial_when_verified_child_survives(monkeypatch, tmp_path):
    project_dir = str(tmp_path / "app")
    root, cmd, npm, next_dev, start_server = _make_dev_tree(project_dir)
    start_server.stubborn = True
    by_pid = {p.pid: p for p in [root, cmd, npm, next_dev, start_server]}

    def fake_process(pid):
        return by_pid[pid]

    def fake_run(args, **kwargs):
        pid = int(args[args.index("/PID") + 1])
        by_pid[pid].mark_taskkill()
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pr, "_IS_WINDOWS", True)
    monkeypatch.setattr(ProcessRegistry, "_host_pid_is_ours", classmethod(lambda cls, pid, expected_start: pid == 1000))
    monkeypatch.setattr(pr.psutil, "Process", fake_process)
    monkeypatch.setattr(pr.subprocess, "run", fake_run)

    result = ProcessRegistry._terminate_host_pid(1000, expected_start=123456)

    assert result["status"] == "partial_kill_children_remain"
    assert result["child_pids"] == [1004]
    assert result["remaining_children"][0]["pid"] == 1004
    assert "start-server.js" in result["remaining_children"][0]["cmdline_fingerprint"]
    assert "cwd_fingerprint" in result["remaining_children"][0]
    assert project_dir not in result["remaining_children"][0]["cmdline_fingerprint"]


def test_cleanup_local_dev_server_verifies_port_owner_and_project_path_before_kill(monkeypatch, tmp_path):
    project_dir = str(tmp_path / "app")
    other_dir = str(tmp_path / "other")
    root, cmd, npm, next_dev, start_server = _make_dev_tree(project_dir)
    start_server._connections = [_fake_listener(3000)]
    unrelated = FakeWindowsProcess(
        2000,
        cmdline=["C:/Program Files/nodejs/node.exe", "node_modules/next/dist/bin/next", "dev"],
        cwd=other_dir,
        create_time=200.0,
        name="node.exe",
    )
    unrelated._connections = [_fake_listener(3000)]
    all_procs = [root, cmd, npm, next_dev, start_server, unrelated]
    by_pid = {p.pid: p for p in all_procs}

    def fake_process_iter(attrs=None):
        return list(all_procs)

    monkeypatch.setattr(pr, "_IS_WINDOWS", True)
    monkeypatch.setattr(pr.psutil, "process_iter", fake_process_iter)

    def fake_run(args, **kwargs):
        pid = int(args[args.index("/PID") + 1])
        by_pid[pid].mark_taskkill()
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pr.subprocess, "run", fake_run)

    result = ProcessRegistry.cleanup_local_dev_server(project_dir, port=3000)

    assert result["status"] == "cleaned"
    assert result["killed_pids"] == [1004]
    assert start_server.alive is False
    assert unrelated.alive is True, "matching command on the same port but another cwd is not owned by this project"
