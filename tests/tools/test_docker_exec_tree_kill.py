"""Docker exec invocations must take their in-container process tree with them.

Issue #84967. ``BaseEnvironment._kill_process()`` calls ``proc.kill()``, which on
the Docker backend kills the host-side ``docker exec`` client and nothing else.
The command's descendants keep running inside the container, get reparented, and
hold pids-cgroup slots; enough timed-out calls and ``pids.max`` is exhausted,
after which even ``date`` hangs.

The Python-level tests below pin the wiring (tagging, scoping, best-effort
teardown). The Linux-only test at the bottom runs the teardown script against a
real reparented process tree, which is the part unit tests cannot honestly
assert.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import pytest

import tools.environments.docker as docker_env


def _make_exec_env():
    """A DockerEnvironment with just enough state for _run_bash/_kill_process."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = "test-container"
    env._docker_exe = "/usr/bin/docker"
    env._init_env_args = []
    env._profile_scoped_passthrough = None
    return env


def _exec_id_args(cmd: list[str]) -> list[str]:
    """Return the values of every ``-e HERMES_EXEC_ID=...`` pair in ``cmd``."""
    return [
        cmd[i + 1].split("=", 1)[1]
        for i, token in enumerate(cmd[:-1])
        if token == "-e" and cmd[i + 1].startswith(f"{docker_env._EXEC_ID_ENV}=")
    ]


class _FakeProc:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


def test_run_bash_stamps_a_unique_exec_id_before_the_container_id(monkeypatch):
    """Every exec carries a marker, and it must sit in the docker-arg section.

    Placed after the container id it would be an argument to the *command*
    rather than to ``docker exec``, and would never reach the process env.
    """
    env = _make_exec_env()
    captured = {}

    def _fake_popen(cmd, stdin_data=None):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(docker_env, "_popen_bash", _fake_popen)

    proc = env._run_bash("echo hi")
    cmd = captured["cmd"]

    ids = _exec_id_args(cmd)
    assert len(ids) == 1, cmd
    assert cmd.index("-e") < cmd.index(env._container_id)
    # The id must be recoverable from the handle — the base class hands the
    # kill path nothing else.
    assert proc._hermes_exec_id == ids[0]


def test_each_invocation_gets_its_own_id(monkeypatch):
    """Scoping requirement: concurrent commands share a persistent container.

    If two execs shared a marker, tearing down a timed-out command would also
    kill an unrelated one still running beside it.
    """
    env = _make_exec_env()
    seen = []

    monkeypatch.setattr(
        docker_env, "_popen_bash",
        lambda cmd, stdin_data=None: seen.append(cmd) or _FakeProc(),
    )

    first = env._run_bash("sleep 1")
    second = env._run_bash("sleep 2")

    assert first._hermes_exec_id != second._hermes_exec_id
    assert _exec_id_args(seen[0]) != _exec_id_args(seen[1])


def test_kill_process_tears_down_the_container_tree(monkeypatch):
    """The host-side kill must be joined by an in-container teardown."""
    env = _make_exec_env()
    calls = []
    monkeypatch.setattr(
        docker_env.subprocess, "run",
        lambda cmd, **kw: calls.append((cmd, kw))
        or subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )

    proc = _FakeProc()
    proc._hermes_exec_id = "deadbeef" * 4
    env._kill_process(proc)

    assert proc.killed, "host-side docker exec client must still be killed"
    assert len(calls) == 1, calls
    cmd, kwargs = calls[0]
    assert cmd[:3] == [env._docker_exe, "exec", env._container_id]
    assert f"{docker_env._EXEC_ID_ENV}={'deadbeef' * 4}" in cmd
    # Must not hang the kill path if the daemon is unresponsive.
    assert kwargs.get("timeout") is not None


@pytest.mark.parametrize(
    "attrs",
    [
        pytest.param({"_hermes_exec_id": None}, id="untagged-handle"),
        pytest.param({"_hermes_exec_id": "abc", "_container_id": None},
                     id="container-already-gone"),
    ],
)
def test_no_teardown_attempted_without_a_target(monkeypatch, attrs):
    """A missing id or container is not an error — there is nothing to kill."""
    env = _make_exec_env()
    if "_container_id" in attrs:
        env._container_id = attrs["_container_id"]
    calls = []
    monkeypatch.setattr(
        docker_env.subprocess, "run", lambda cmd, **kw: calls.append(cmd),
    )

    proc = _FakeProc()
    proc._hermes_exec_id = attrs["_hermes_exec_id"]
    env._kill_process(proc)

    assert proc.killed
    assert calls == []


def test_teardown_failure_never_propagates(monkeypatch):
    """Cleanup runs on the timeout/interrupt/exception paths.

    A failure here must not mask the condition that triggered the kill.
    """
    env = _make_exec_env()

    def _boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(docker_env.subprocess, "run", _boom)

    proc = _FakeProc()
    proc._hermes_exec_id = "f" * 32
    env._kill_process(proc)  # must not raise

    assert proc.killed


# ---------------------------------------------------------------------------
# The real thing: run the teardown script against an actual process tree.
# ---------------------------------------------------------------------------

def _run_tree_kill(exec_id: str, grace: str = "1") -> None:
    """Invoke the teardown script for ``exec_id``.

    Also used for fixture cleanup, deliberately: signalling through the script
    keeps these tests clear of ``os.kill``, which the conftest live-system guard
    blocks for anything outside the test subtree. Reaching for the documented
    bypass marker instead would switch that protection off for the whole test.
    """
    subprocess.run(
        [
            "bash", "-c", docker_env._TREE_KILL_SCRIPT,
            "hermes-tree-kill",
            f"{docker_env._EXEC_ID_ENV}={exec_id}",
            grace,
        ],
        capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
    )


def _wait_until_gone(exec_id: str, timeout: float = 10) -> list[int]:
    deadline = time.monotonic() + timeout
    while _marked_pids(exec_id) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _marked_pids(exec_id)


def _marked_pids(exec_id: str) -> list[int]:
    """PIDs whose environment carries ``exec_id`` — the survivor check."""
    marker = f"{docker_env._EXEC_ID_ENV}={exec_id}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as handle:
                if marker in handle.read().split(b"\0"):
                    found.append(int(name))
        except (OSError, ValueError):
            continue
    return found


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="needs /proc and POSIX signals; the container side is always Linux",
)
def test_tree_kill_script_reaps_reparented_descendants():
    """A descendant that outlives its launcher must still be found and killed.

    This is the shape the leak actually takes: the launcher exits (or is killed
    host-side), its children are adopted by PID 1, and any teardown that walks
    parent PIDs has already lost them. They keep their environment, which is
    why the script matches on that instead.
    """
    exec_id = uuid.uuid4().hex
    child_env = os.environ.copy()
    child_env[docker_env._EXEC_ID_ENV] = exec_id

    # The launcher backgrounds a sleep and exits immediately, so the survivor
    # is reparented away before the teardown ever runs. ``echo $!`` runs after
    # the fork, so reading it makes the handoff deterministic rather than
    # polling and hoping the child has appeared yet.
    launcher = subprocess.Popen(
        ["bash", "-c", "sleep 600 & echo $!; exit 0"],
        env=child_env, stdout=subprocess.PIPE, text=True,
    )
    child_pid = int(launcher.stdout.readline().strip())
    launcher.wait(timeout=10)

    assert child_pid not in (launcher.pid, os.getpid())
    survivors = _marked_pids(exec_id)
    assert child_pid in survivors, (
        f"fixture failed to leave a reparented descendant: {survivors}"
    )

    try:
        _run_tree_kill(exec_id)
        assert not _wait_until_gone(exec_id), (
            f"orphans survived teardown: {_marked_pids(exec_id)}"
        )
    finally:
        _run_tree_kill(exec_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="needs /proc and POSIX signals; the container side is always Linux",
)
def test_tree_kill_script_leaves_unrelated_processes_alone():
    """Point 4 of the issue: a shared persistent container must stay usable."""
    target_id = uuid.uuid4().hex
    bystander_id = uuid.uuid4().hex

    def _spawn(exec_id):
        child_env = os.environ.copy()
        child_env[docker_env._EXEC_ID_ENV] = exec_id
        return subprocess.Popen(["bash", "-c", "sleep 600"], env=child_env)

    target = _spawn(target_id)
    bystander = _spawn(bystander_id)
    try:
        _run_tree_kill(target_id)

        assert not _wait_until_gone(target_id), (
            "tagged process should have been killed"
        )
        assert _marked_pids(bystander_id), "untagged process must be untouched"
        assert bystander.poll() is None
    finally:
        for exec_id in (target_id, bystander_id):
            _run_tree_kill(exec_id)
        for proc in (target, bystander):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
