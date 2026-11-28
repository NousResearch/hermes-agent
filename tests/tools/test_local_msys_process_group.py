"""MSYS group selection and failure handling do not depend on the host OS."""

import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import psutil

from tools.environments import local


HEADER = "PID PPID PGID WINPID TTY UID STIME COMMAND\n"
COMMAND = "100 1 100 1000 ? 42 10:00 /usr/bin/bash\n"
PIPELINE = "101 100 100 1001 ? 42 10:00 /usr/bin/sleep\n"
HOST = "200 1 200 2000 ? 42 10:00 /usr/bin/python\n"


@pytest.mark.parametrize("output, expected", [
    (HEADER + COMMAND + PIPELINE + HOST, 100),
    ("\n  PGID\tWINPID  PPID PID COMMAND\n 100\t1000 1 100 /usr/bin/bash\n"
     " 100 1001 100 101 /usr/bin/sleep\n", 100),
    (HEADER + COMMAND + "I " + PIPELINE, 100),
    (HEADER + HOST, None),
    (HEADER + COMMAND + COMMAND, None),
    (HEADER + COMMAND + "102 1 102 1000 ? 42 10:00 /usr/bin/bash\n", None),
    (HEADER + COMMAND.replace("100 1 100", "100 1 0"), None),
    (HEADER + COMMAND.replace("100 1 100", "100 1 1"), None),
    (HEADER + COMMAND.replace("100 1 100", "100 1 -100"), None),
    (HEADER + COMMAND.replace("100 1 100", "100 1 99"), None),
    (HEADER + COMMAND + HOST.replace("200 1 200", "200 1 100"), None),
    (HEADER + COMMAND + PIPELINE.replace("101 100", "101 200"), 100),
    (HEADER + COMMAND + PIPELINE.replace("101 100", "101 1"), 100),
    (HEADER + COMMAND + "101 102 100 1001 ? 42 10:00 /usr/bin/cat\n"
     "102 101 100 1002 ? 42 10:00 /usr/bin/cat\n", 100),
    (HEADER + COMMAND + "101 100 100 1001 ? 42 10:00 /usr/bin/cat\n"
     "102 101 100 1002 ? 42 10:00 /usr/bin/sleep\n", 100),
    ("PID PPID PGID COMMAND\n100 1 100 /usr/bin/bash\n", None),
    ("PID PPID PGID WINPID WINPID\n100 1 100 1000 1000\n", None),
    (HEADER + "not a process row\n" + COMMAND, None),
    (HEADER + COMMAND + "101 100 bad 1001 ? 42 10:00 /usr/bin/cat\n", None),
    ("", None),
])
def test_select_only_unambiguous_command_group(output, expected):
    from tools.environments.local_windows_msys import _command_pgid

    assert _command_pgid(output, {1000: 0}, 2000) == expected


def test_never_select_hermes_itself():
    from tools.environments.local_windows_msys import _command_pgid

    assert _command_pgid(HEADER + COMMAND, {1000: 0}, 1000) is None


@pytest.mark.parametrize("output, candidates, expected", [
    (HEADER + COMMAND + PIPELINE, {1000: 0, 1001: 1}, 100),
    (HEADER + COMMAND + PIPELINE, {999: 0, 1000: 1}, 100),
    (HEADER + COMMAND + PIPELINE, {999: 0, 1001: 1}, None),
    (HEADER + COMMAND.replace("/usr/bin/bash", "/usr/bin/sh"), {1000: 0}, 100),
    (HEADER + COMMAND.replace("/usr/bin/bash", "/usr/bin/cat"), {1000: 0}, 100),
    (HEADER + COMMAND.replace("/usr/bin/bash", r"C:\Git\usr\bin\bash.exe"), {1000: 0}, 100),
    (HEADER + COMMAND.replace("/usr/bin/bash", r"C:\Program Files\Git\usr\bin\bash.exe"), {1000: 0}, 100),
    (HEADER + "1000 0 0 1000 ? 0 10:00 C:\\Git\\bin\\bash.exe\n" +
     COMMAND.replace("1000", "1001"), {1000: 0, 1001: 1}, 100),
    (HEADER + COMMAND + "102 1 102 1002 ? 42 10:00 /usr/bin/bash\n", {1000: 1, 1002: 1}, None),
    (HEADER + COMMAND + "102 100 102 1002 ? 42 10:00 /usr/bin/bash\n", {1000: 0, 1002: 1}, 100),
    (HEADER + COMMAND + "102 100 102 1002 ? 42 10:00 /usr/bin/bash\n", {1000: 1, 1002: 2}, 100),
    (HEADER + COMMAND + "102 100 102 1002 ? 42 10:00 /usr/bin/bash\n", {1000: 3, 1002: 2}, 102),
    (HEADER + COMMAND + "102 100 100 1002 ? 42 10:00 /usr/bin/bash\n" +
     "103 1 100 1003 ? 42 10:00 /usr/bin/sleep\n", {1000: 0, 1002: 1}, 100),
    (HEADER + COMMAND + "102 1 102 1002 ? 42 10:00 /usr/bin/sh\n", {999: 0, 1002: 2}, 102),
    (HEADER + COMMAND, {}, None),
    (HEADER + COMMAND, {1000: 0, 2000: 1}, None),
    ("PID PPID PGID WINPID\n100 1 100 1000\n", {1000: 0}, None),
])
def test_group_candidates_select_nearest_owned_leader(output, candidates, expected):
    from tools.environments.local_windows_msys import _command_pgid

    assert _command_pgid(output, candidates, 2000) == expected


@pytest.fixture
def msys_command(tmp_path, monkeypatch):
    from tools.environments import local_windows_msys as msys

    bin_dir = tmp_path / "Git" / "usr" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("bash.exe", "ps.exe", "kill.exe"):
        (bin_dir / name).touch()
    monkeypatch.setattr(local, "_find_bash", lambda: str(bin_dir / "bash.exe"))
    monkeypatch.setattr(msys.os, "getpid", lambda: 2000)
    monkeypatch.setattr(msys, "windows_hide_flags", lambda: 0x08000000)
    run = Mock(side_effect=[SimpleNamespace(stdout=HEADER + COMMAND + PIPELINE), SimpleNamespace()])
    monkeypatch.setattr(msys.subprocess, "run", run)
    proc = SimpleNamespace(pid=1000, poll=Mock(return_value=None),
                           args=[str(bin_dir / "bash.exe"), "-c", "sleep 60 | cat"])
    process = Mock()
    process.children.return_value = []
    process.create_time.return_value = 10.0
    process_lookup = Mock(return_value=process)
    monkeypatch.setattr("psutil.Process", process_lookup)
    return SimpleNamespace(msys=msys, bin_dir=bin_dir, proc=proc, run=run,
                           process=process, process_lookup=process_lookup)


@pytest.mark.parametrize("bash_pid", [1000, 1001])
def test_bash_can_be_the_launcher_or_its_direct_child(msys_command, bash_pid):
    w = msys_command
    w.process.children.return_value = [SimpleNamespace(pid=1001, ppid=lambda: 1000, create_time=lambda: 10.0)]
    w.run.side_effect = [SimpleNamespace(stdout=HEADER + COMMAND.replace("1000", str(bash_pid))),
                         SimpleNamespace()]
    group = w.msys.find_msys_process_group(w.proc)
    assert w.process_lookup.call_args_list[0].args == (1000,)
    w.process.children.assert_called_once_with(recursive=True)
    assert group == (w.bin_dir / "kill.exe", 100)
    assert w.run.call_count == 1


def test_ambiguous_direct_children_fall_back_without_a_group_signal(msys_command):
    w = msys_command
    w.process.children.return_value = [SimpleNamespace(pid=pid, ppid=lambda: 1000, create_time=lambda: 10.0)
                                     for pid in (1001, 1002)]
    w.run.side_effect = [SimpleNamespace(stdout=HEADER +
        COMMAND.replace("1000", "1001") + "102 1 102 1002 ? 42 10:00 /usr/bin/bash\n")]
    group = w.msys.find_msys_process_group(w.proc)
    assert group is None
    assert w.run.call_count == 1


def test_recursive_windows_descendants_choose_shallowest_leader(msys_command):
    w = msys_command
    # Enumeration order is not a depth ordering; both shells lead groups.
    w.process.children.return_value = [
        SimpleNamespace(pid=1002, ppid=lambda: 1001, create_time=lambda: 10.0),
        SimpleNamespace(pid=1001, ppid=lambda: 1000, create_time=lambda: 10.0),
    ]
    w.run.side_effect = [SimpleNamespace(stdout=HEADER + COMMAND.replace("1000", "1001") +
        "102 100 102 1002 ? 42 10:00 /usr/bin/bash\n")]
    assert w.msys.find_msys_process_group(w.proc) == (w.bin_dir / "kill.exe", 100)
    w.process.children.assert_called_once_with(recursive=True)


def test_a_group_leader_can_be_a_deep_windows_descendant(msys_command):
    w = msys_command
    w.process.children.return_value = [
        SimpleNamespace(pid=1002, ppid=lambda: 1001, create_time=lambda: 10.0),
        SimpleNamespace(pid=1001, ppid=lambda: 1000, create_time=lambda: 10.0),
    ]
    w.run.side_effect = [SimpleNamespace(stdout=HEADER + COMMAND.replace("1000", "1002"))]
    assert w.msys.find_msys_process_group(w.proc) == (w.bin_dir / "kill.exe", 100)


def test_pid_reused_while_finding_children_is_not_signalled(msys_command):
    w = msys_command
    # poll reads Popen's original handle, not a new handle for the recycled PID.
    w.process.children.side_effect = lambda **kwargs: setattr(w.proc.poll, "return_value", 0) or []
    group = w.msys.find_msys_process_group(w.proc)
    assert group is None
    w.run.assert_not_called()


@pytest.mark.parametrize("current_identity", [20.0, psutil.NoSuchProcess(1001), psutil.AccessDenied(1001)])
def test_stale_descendant_identity_is_rejected_after_ps(msys_command, current_identity):
    w = msys_command
    w.process.children.return_value = [SimpleNamespace(pid=1001, ppid=lambda: 1000, create_time=lambda: 10.0)]
    w.run.side_effect = [SimpleNamespace(stdout=HEADER + COMMAND.replace("1000", "1001"))]
    current_process = Mock()
    current_process.create_time.side_effect = [current_identity]
    w.process_lookup.side_effect = lambda pid: w.process if pid == 1000 else current_process
    assert w.msys.find_msys_process_group(w.proc) is None
    assert w.run.call_count == 1


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("layout", ["bin", "usr/bin"])
def test_teardown_tools_come_from_the_started_command(msys_command, monkeypatch, layout):
    w = msys_command
    root = w.bin_dir.parent.parent
    w.proc.args[0] = str(root / layout / "bash.exe")
    find_bash = Mock(side_effect=AssertionError("teardown must not discover or probe shells"))
    monkeypatch.setattr(local, "_find_bash", find_bash)
    fallback = Mock(side_effect=lambda proc: setattr(proc.poll, "return_value", 0))
    monkeypatch.setattr(local, "_kill_process_windows", fallback)
    object.__new__(local.LocalEnvironment)._kill_process(w.proc)
    find_bash.assert_not_called()
    fallback.assert_called_once_with(w.proc)
    assert [call.args[0] for call in w.run.call_args_list] == [
        [str(w.bin_dir / "ps.exe"), "-W", "-l"],
        [str(w.bin_dir / "kill.exe"), "-KILL", "--", "-100"],
    ]
    for call in w.run.call_args_list:
        assert call.kwargs == dict(capture_output=True, encoding="utf-8", errors="replace",
                                   creationflags=0x08000000, timeout=2, check=True)


@pytest.mark.parametrize("missing", ["ps.exe", "kill.exe"])
def test_missing_utilities_are_a_noop(msys_command, missing):
    w = msys_command
    (w.bin_dir / missing).unlink()
    group = w.msys.find_msys_process_group(w.proc)
    assert group is None
    w.run.assert_not_called()


@pytest.mark.parametrize("failure", [
    FileNotFoundError("missing utility"),
    subprocess.TimeoutExpired("ps", 2),
    subprocess.CalledProcessError(1, "ps"),
    RuntimeError("helper failed"),
])
@pytest.mark.parametrize("stage", ["lookup", "ps", "kill"])
def test_helper_failures_are_swallowed_once(msys_command, monkeypatch, caplog, failure, stage):
    w = msys_command
    if stage == "lookup":
        w.process_lookup.side_effect = failure
    elif stage == "ps":
        w.run.side_effect = failure
    else:
        w.run.side_effect = [SimpleNamespace(stdout=HEADER + COMMAND), failure]
    with caplog.at_level(logging.DEBUG, logger=w.msys.__name__):
        group = w.msys.find_msys_process_group(w.proc)
        if group is not None:
            w.msys.kill_msys_process_group(group)
    assert len(caplog.records) == 1
    assert "MSYS" in caplog.records[0].message


def test_unsafe_group_does_not_invoke_kill(msys_command):
    w = msys_command
    w.run.side_effect = [SimpleNamespace(stdout=HEADER + HOST)]
    group = w.msys.find_msys_process_group(w.proc)
    assert group is None
    assert w.run.call_count == 1


@pytest.mark.parametrize("exited_on_poll, helper_calls", [(1, 0), (2, 0), (3, 1)])
def test_an_exited_command_is_not_signalled(msys_command, exited_on_poll, helper_calls):
    w = msys_command
    w.proc.poll.side_effect = [None] * (exited_on_poll - 1) + [0]
    group = w.msys.find_msys_process_group(w.proc)
    assert group is None
    assert w.run.call_count == helper_calls


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("method", ["_kill_process", "_force_kill_process"])
@pytest.mark.parametrize("failure", [None, "missing-ps", "missing-kill", "ps-timeout", "kill-failure", "no-group", "fallback-error"])
def test_lookup_precedes_windows_fallback_and_group_signal_follows_it(msys_command, monkeypatch, method, failure):
    w = msys_command
    events = []
    find_bash = Mock(side_effect=AssertionError("teardown must not discover or probe shells"))
    monkeypatch.setattr(local, "_find_bash", find_bash)
    if failure in ("missing-ps", "missing-kill"):
        (w.bin_dir / ("ps.exe" if failure == "missing-ps" else "kill.exe")).unlink()

    def run(argv, **kwargs):
        stage = "ps" if Path(argv[0]).name == "ps.exe" else "kill"
        events.append(stage)
        if failure == "ps-timeout":
            raise subprocess.TimeoutExpired(argv, 2)
        if stage == "kill" and failure == "kill-failure":
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout=HEADER + (HOST if failure == "no-group" else COMMAND))

    w.run.side_effect = run

    def fallback(proc):
        events.append("fallback")
        proc.poll.return_value = 0
        if failure == "fallback-error":
            raise OSError("taskkill failed")

    w.proc.kill = Mock()
    monkeypatch.setattr(local, "_kill_process_windows", fallback)
    getattr(object.__new__(local.LocalEnvironment), method)(w.proc)
    expected = {None: ["ps", "fallback", "kill"], "missing-ps": ["fallback"], "missing-kill": ["fallback"],
                "ps-timeout": ["ps", "fallback"], "kill-failure": ["ps", "fallback", "kill"],
                "no-group": ["ps", "fallback"], "fallback-error": ["ps", "fallback", "kill"]}
    assert events == expected[failure]
    find_bash.assert_not_called()
    if failure in ("missing-ps", "missing-kill"):
        w.run.assert_not_called()
    assert w.proc.kill.call_count == (1 if failure == "fallback-error" else 0)


@pytest.mark.platforms("posix")
def test_posix_still_uses_only_its_existing_group_teardown(monkeypatch):
    from tools.environments import local_windows_msys as msys

    group_kill = Mock()
    msys_lookup = Mock()
    msys_kill = Mock()
    monkeypatch.setattr(local, "_kill_process_group_posix", group_kill)
    monkeypatch.setattr(local, "find_msys_process_group", msys_lookup)
    monkeypatch.setattr(local, "kill_msys_process_group", msys_kill)
    proc = SimpleNamespace(pid=1000)
    object.__new__(local.LocalEnvironment)._kill_process(proc)
    group_kill.assert_called_once_with(proc)
    msys_lookup.assert_not_called()
    msys_kill.assert_not_called()
