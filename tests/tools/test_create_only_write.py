import json

import pytest

from tools import file_tools as ft
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def ops(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    backend = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": backend)
    return backend


@pytest.mark.platforms("posix")
def test_create_only_dispatch_refuses_existing_entries(ops, tmp_path):
    target = tmp_path / "original.txt"
    target.write_bytes(b"original\n")
    empty = tmp_path / "empty.txt"
    empty.touch()
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    dangling = tmp_path / "dangling.txt"
    dangling.symlink_to(tmp_path / "absent.txt")
    directory = tmp_path / "directory"
    directory.mkdir()
    for entry in (target, empty, link, dangling, directory):
        out = json.loads(registry.dispatch("write_file", {
            "path": str(entry), "content": "replacement\n", "create_only": True}, task_id=str(entry)))
        assert out.get("status") == "already_exists", out
    assert target.read_bytes() == b"original\n"
    assert empty.read_bytes() == b""
    assert link.is_symlink() and dangling.is_symlink() and directory.is_dir()
    assert not (tmp_path / "absent.txt").exists()


def test_concurrent_create_only_publishes_one_complete_file(ops, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    target = tmp_path / "nested" / "new.txt"
    candidates = [f"candidate-é-{i}\n" * 1000 for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda content: ops.write_file(str(target), content, create_only=True), candidates))
    winners = [i for i, result in enumerate(results) if not result.error]
    assert len(winners) == 1, results
    assert target.read_bytes() == candidates[winners[0]].encode()
    assert all(result.status == "already_exists" for result in results if result.error)
    assert not list(target.parent.glob(".hermes-create-*"))


@pytest.fixture
def remote_ops(ops, monkeypatch):
    """A backend that is not the host: create_only cannot use the agent's interpreter."""
    monkeypatch.setattr(ops, "_lsp_local_only", lambda: False)
    return ops


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("python", ["hermes-missing-python3", None])
def test_remote_create_only_with_and_without_python3(remote_ops, tmp_path, monkeypatch, python):
    import sys

    monkeypatch.setattr(remote_ops, "_python_interpreter_cmd", lambda: python or sys.executable)
    target = tmp_path / "nested" / "new.txt"
    payload = "a\r\nb é"
    result = remote_ops.write_file(str(target), payload, create_only=True)
    assert result.error is None and result.verified is True, result
    assert target.read_bytes() == payload.encode()
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    for entry in (target, dangling, tmp_path / "nested"):
        refused = remote_ops.write_file(str(entry), "replacement", create_only=True)
        assert refused.status == "already_exists", refused
    assert target.read_bytes() == payload.encode() and not (tmp_path / "absent").exists()
    assert not list(tmp_path.rglob(".hermes-create-*"))


def test_write_file_description_scopes_overwrite_to_the_default():
    desc = ft.WRITE_FILE_SCHEMA["description"]
    overwrite = desc.index("OVERWRITES the entire file by default")
    assert desc.index("create_only=true instead refuses any existing path") > overwrite
    assert desc.count("create_only") == 1


def test_write_file_schema_has_no_typographic_apostrophes():
    assert "\u2019" not in json.dumps(ft.WRITE_FILE_SCHEMA, ensure_ascii=False)


def _create_command(remote_ops, monkeypatch, path, python):
    """The exact remote-backend command atomic_create would hand to the backend."""
    from tools.file_operations_common import ExecuteResult
    from tools.file_operations_create import atomic_create

    seen = []
    monkeypatch.setattr(remote_ops, "_python_interpreter_cmd", lambda: python)
    monkeypatch.setattr(remote_ops, "_exec", lambda command, **kw: seen.append(command) or ExecuteResult(stdout="", exit_code=0))
    atomic_create(remote_ops, str(path), "unused")
    return seen[0]


def _session_commands(sid):
    import os

    names = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
            with open(f"/proc/{pid}/comm") as comm:
                name = comm.read().strip()
        except OSError:
            continue
        if int(fields[3]) == sid and fields[0] != "Z":
            names.append(name)
    return names


_SHELLS = {"dash": ["dash", "-c"], "bash": ["bash", "-c"], "busybox": ["busybox", "sh", "-c"]}


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP", "SIGINT"])
@pytest.mark.parametrize("lane,shell", [("shell", "dash"), ("shell", "bash"), ("shell", "busybox"),
                                        ("python", "dash"), ("python", "bash")])
def test_interrupted_create_only_leaves_no_temp_file(remote_ops, tmp_path, monkeypatch, lane, shell, signame):
    """A catchable signal mid-write removes the temp file and exits nonzero, on every shell."""
    import os
    import shutil
    import signal
    import subprocess
    import sys

    from tests.waiting import wait_until

    argv = _SHELLS[shell]
    if not shutil.which(argv[0]):
        pytest.skip(f"{argv[0]} not installed")
    target = tmp_path / "out" / "new.txt"
    command = _create_command(remote_ops, monkeypatch, target, sys.executable if lane == "python" else "hermes-missing-python3")
    proc = subprocess.Popen([*argv, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=tmp_path, start_new_session=True)
    try:
        proc.stdin.write(b"partial")
        proc.stdin.flush()
        wait_until(lambda: list(target.parent.glob(".hermes-create-*")), what="the temp file")
        if lane == "shell":  # the writer is blocked reading stdin
            wait_until(lambda: "cat" in _session_commands(proc.pid), what="cat to start")
        os.killpg(proc.pid, getattr(signal, signame))
        proc.wait(timeout=30)
        wait_until(lambda: not _session_commands(proc.pid), what="the session to exit")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.stdin.close()
        proc.stdout.close()
    assert proc.returncode != 0
    assert not target.exists()
    assert not list(target.parent.glob(".hermes-create-*"))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("python", ["hermes-missing-python3", None])
@pytest.mark.parametrize("parent", ["dir\n", "dir\n\n", "-dir"])
def test_remote_create_only_keeps_the_parent_path_exact(remote_ops, tmp_path, monkeypatch, python, parent):
    import sys

    from tools.file_operations_create import atomic_create

    monkeypatch.setattr(remote_ops, "_python_interpreter_cmd", lambda: python or sys.executable)
    target = tmp_path / parent / "new.txt"
    result = atomic_create(remote_ops, str(target), "body\n")
    assert result.exit_code == 0, result
    assert target.read_bytes() == b"body\n"
    assert not (tmp_path / "dir").exists()
    assert not list(tmp_path.rglob(".hermes-create-*"))
