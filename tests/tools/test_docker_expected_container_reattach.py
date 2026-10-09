"""A caller that already knows which container it wants (Bot Desktop's sandbox.json marker)
must not be defeated by the normal label search missing it -- a persistent container's reuse
key can legitimately drift from what the current process would compute (a profile migrating
from a standalone to a multiplexed gateway changes persistent Docker's task-id key; #t_2ca05370).

``expected_container_id`` lets ``DockerEnvironment`` try that exact container FIRST, falling
back to the normal label search (and ultimately a fresh container) only when it is gone or
fails its own safety check (still a hermes container, still labeled for this profile).
"""
from __future__ import annotations

import subprocess

from tools.environments import docker as docker_env


def _make_dummy_env(monkeypatch, **kwargs):
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: kwargs.get("profile", "wian"))
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        task_id=kwargs.get("task_id", "profile:wian"),
        persistent_filesystem=False,
        persist_across_processes=True,
        expected_container_id=kwargs.get("expected_container_id"),
    )


def _mock_docker(monkeypatch, *, inspect_stdout: str, inspect_returncode: int = 0):
    """``docker run``/``version`` succeed as usual; ``docker inspect <id> --format ...`` for the
    expected-container probe (identified by ``hermes-agent`` in its format string) answers with
    *inspect_stdout*; the image/network reconciliation probes report a clean match so only the
    expected-container branch is under test. A label-search ``docker ps`` always misses
    (simulating a drifted reuse key) so reuse can ONLY happen via the expected-id hint."""
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        argv = list(cmd) if isinstance(cmd, list) else cmd
        calls.append(argv)
        joined = " ".join(argv)
        if len(argv) >= 2 and argv[1] == "version":
            return subprocess.CompletedProcess(argv, 0, stdout="Docker version", stderr="")
        if len(argv) >= 2 and argv[1] == "ps":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")  # label search: no match
        if len(argv) >= 2 and argv[1] == "inspect" and "hermes-agent" in joined:
            return subprocess.CompletedProcess(argv, inspect_returncode, stdout=inspect_stdout, stderr="")
        if len(argv) >= 2 and argv[1] == "inspect" and "Config.Image" in joined:
            return subprocess.CompletedProcess(argv, 0, stdout="python:3.11\n", stderr="")
        if len(argv) >= 2 and argv[1] == "inspect" and "NetworkMode" in joined:
            return subprocess.CompletedProcess(argv, 0, stdout="default\n", stderr="")
        if len(argv) >= 2 and argv[1] == "run":
            return subprocess.CompletedProcess(argv, 0, stdout="freshly-created-id\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env._cgroup_limits_ok = True
    return calls


def test_reattaches_to_the_expected_container_when_the_label_search_misses(monkeypatch):
    """The historical bug: a profile's persistent-Docker reuse key drifted (standalone ->
    multiplexed gateway, or any other config/session-context change), so ``docker ps`` with the
    CURRENT labels finds nothing even though the real container is alive and correctly labeled
    for this profile. The expected-id hint must still find and attach to it."""
    calls = _mock_docker(monkeypatch, inspect_stdout="running\t1\twian\n")

    env = _make_dummy_env(monkeypatch, expected_container_id="c0ffee00")

    assert env._container_id == "c0ffee00"
    assert not any(len(c) >= 2 and c[1] == "run" for c in calls), \
        "a fresh container must not be created when the expected one reattaches"


def test_falls_back_to_a_fresh_container_when_the_expected_one_is_gone(monkeypatch):
    """A marker naming a container that no longer exists (removed out from under us) must not
    wedge creation -- it falls through to the normal label search and then a fresh container,
    exactly as if no hint had been given."""
    calls = _mock_docker(monkeypatch, inspect_stdout="", inspect_returncode=1)

    env = _make_dummy_env(monkeypatch, expected_container_id="deadbeef")

    assert env._container_id == "freshly-created-id"
    assert any(len(c) >= 2 and c[1] == "run" for c in calls)


def test_refuses_to_adopt_a_container_labeled_for_a_different_profile(monkeypatch):
    """A safety net on the hint itself: even a caller-supplied id is only trusted after
    confirming it still carries OUR OWN hermes-agent/hermes-profile labels, so a stale or
    cross-profile id can never be force-attached."""
    _mock_docker(monkeypatch, inspect_stdout="running\t1\tsome-other-profile\n")

    env = _make_dummy_env(monkeypatch, expected_container_id="c0ffee00", task_id="profile:wian")

    assert env._container_id == "freshly-created-id"


def test_no_hint_is_a_pure_no_op():
    """``expected_container_id=None`` (the default) never calls the verification probe."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._expected_container_id = None
    env._docker_exe = "docker"
    assert env._verified_expected_container("wian") is None
