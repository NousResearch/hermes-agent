"""Cross-profile bind-mount boundary for Docker containers (#101132).

A multiplexed gateway serves every profile from one interpreter. The container
label ``hermes-profile`` and every Hermes-managed mount follow the per-turn
profile scope, but ``terminal.docker_volumes`` and the
``docker_mount_cwd_to_workspace`` workspace bind reach the container through
process-global ``TERMINAL_*`` env vars that are written once per process. A
container created for profile A could therefore be handed profile B's
directories — read-write, since those are the only writable user mounts.

These tests pin the boundary itself: whatever produced the source path, a
container serving one profile never receives another profile's Hermes home.
"""

import subprocess

import pytest

from tools.environments import docker as docker_env


def _mock_docker(monkeypatch):
    """Capture ``docker run`` argv without touching a real daemon."""
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, list) else cmd)
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


@pytest.fixture
def two_profiles(monkeypatch, tmp_path):
    """A Hermes root with ``alice`` and ``bob``, scoped to ``bob``'s turn."""
    root = tmp_path / ".hermes"
    homes = {"default": root}
    for name in ("alice", "bob"):
        home = root / "profiles" / name
        (home / "cache" / "documents").mkdir(parents=True)
        homes[name] = home

    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "bob")
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: homes["bob"])
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    return homes


def _run_argv(calls):
    for cmd in calls:
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run":
            return " ".join(cmd)
    raise AssertionError("docker run was never called")


def _make_env(**kwargs):
    return docker_env.DockerEnvironment(
        image="python:3.11",
        cwd=kwargs.pop("cwd", "/root"),
        timeout=60,
        task_id="test-task",
        volumes=kwargs.pop("volumes", []),
        **kwargs,
    )


# --------------------------------------------------------------------------
# docker_volumes
# --------------------------------------------------------------------------

def test_foreign_profile_volume_is_dropped(monkeypatch, two_profiles, caplog):
    """Bob's container must not bind-mount Alice's documents cache."""
    calls = _mock_docker(monkeypatch)
    foreign = two_profiles["alice"] / "cache" / "documents"

    with caplog.at_level("ERROR"):
        _make_env(volumes=[f"{foreign.as_posix()}:/output"])

    argv = _run_argv(calls)
    assert "/output" not in argv
    assert "alice" not in argv
    assert "belongs to profile 'alice'" in caplog.text


def test_own_profile_volume_still_mounts(monkeypatch, two_profiles):
    """The profile's own directory is exactly what the config intends."""
    calls = _mock_docker(monkeypatch)
    own = two_profiles["bob"] / "cache" / "documents"

    _make_env(volumes=[f"{own.as_posix()}:/output"])

    assert f"{own.as_posix()}:/output" in _run_argv(calls)


def test_unrelated_host_volume_still_mounts(monkeypatch, two_profiles, tmp_path):
    """Paths outside every Hermes home are the operator's business, not ours."""
    calls = _mock_docker(monkeypatch)
    shared = tmp_path / "shared-data"
    shared.mkdir()

    _make_env(volumes=[f"{shared.as_posix()}:/data"])

    assert f"{shared.as_posix()}:/data" in _run_argv(calls)


def test_default_root_volume_is_dropped_for_named_profile(monkeypatch, two_profiles):
    """``~/.hermes`` is the default profile's home — foreign to ``bob``."""
    calls = _mock_docker(monkeypatch)

    _make_env(volumes=[f"{two_profiles['default'].as_posix()}:/hermes"])

    assert "/hermes" not in _run_argv(calls)


def test_profiles_tree_volume_is_dropped(monkeypatch, two_profiles):
    """A source CONTAINING foreign homes exposes every profile at once."""
    calls = _mock_docker(monkeypatch)
    profiles_root = two_profiles["default"] / "profiles"

    _make_env(volumes=[f"{profiles_root.as_posix()}:/all"])

    assert "/all" not in _run_argv(calls)


def test_named_volume_is_untouched(monkeypatch, two_profiles):
    """A named Docker volume has no host path and cannot cross a boundary."""
    calls = _mock_docker(monkeypatch)

    _make_env(volumes=["cache-vol:/cache"])

    assert "cache-vol:/cache" in _run_argv(calls)


def test_shared_container_key_opts_out(monkeypatch, two_profiles):
    """Explicit container sharing has no single owning profile to protect."""
    calls = _mock_docker(monkeypatch)
    foreign = two_profiles["alice"] / "cache" / "documents"

    _make_env(
        volumes=[f"{foreign.as_posix()}:/output"],
        shared_container_key="family",
    )

    assert f"{foreign.as_posix()}:/output" in _run_argv(calls)


# --------------------------------------------------------------------------
# docker_mount_cwd_to_workspace
# --------------------------------------------------------------------------

def test_foreign_profile_cwd_mount_is_dropped(monkeypatch, two_profiles):
    """The frozen ``TERMINAL_CWD`` must not put Alice's home at /workspace."""
    calls = _mock_docker(monkeypatch)
    foreign_cwd = two_profiles["alice"] / "workspace"
    foreign_cwd.mkdir()

    _make_env(
        cwd="/workspace",
        host_cwd=str(foreign_cwd),
        auto_mount_cwd=True,
    )

    argv = _run_argv(calls)
    assert f"{foreign_cwd}:/workspace" not in argv
    assert "alice" not in argv


def test_own_profile_cwd_mount_survives(monkeypatch, two_profiles):
    """Bob's own workspace still binds — the guard is not a blanket ban."""
    calls = _mock_docker(monkeypatch)
    own_cwd = two_profiles["bob"] / "workspace"
    own_cwd.mkdir()

    _make_env(
        cwd="/workspace",
        host_cwd=str(own_cwd),
        auto_mount_cwd=True,
    )

    assert f"{own_cwd}:/workspace" in _run_argv(calls)


def test_dropped_cwd_mount_falls_back_to_profile_sandbox(monkeypatch, two_profiles, tmp_path):
    """Refusing the bind leaves the persistent per-profile sandbox mounted."""
    calls = _mock_docker(monkeypatch)
    foreign_cwd = two_profiles["alice"] / "workspace"
    foreign_cwd.mkdir()
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setattr(
        "tools.environments.base.get_sandbox_dir", lambda: sandbox_root
    )

    _make_env(
        cwd="/workspace",
        host_cwd=str(foreign_cwd),
        auto_mount_cwd=True,
        persistent_filesystem=True,
    )

    argv = _run_argv(calls)
    assert ":/workspace" in argv
    assert str(foreign_cwd) not in argv
    assert str(sandbox_root) in argv


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("/host/dir:/out", "/host/dir"),
        ("/host/dir:/out:ro", "/host/dir"),
        ("C:/host/dir:/out", "C:/host/dir"),
        ("C:\\host\\dir:/out", "C:\\host\\dir"),
        ("~/dir:/out", "~/dir"),
        ("./rel:/out", "./rel"),
        ("named:/out", ""),
        ("no-colon", ""),
        ("", ""),
    ],
)
def test_volume_host_source_parsing(spec, expected):
    assert docker_env._docker_volume_host_source(spec) == expected


def test_probe_failure_is_permissive(monkeypatch, two_profiles):
    """An unreadable profiles tree must never make containers unstartable."""
    import hermes_constants

    def _boom():
        raise OSError("profiles unreadable")

    monkeypatch.setattr(hermes_constants, "get_hermes_home", _boom)
    assert docker_env._foreign_profile_mount_owner("/anything") is None


# --------------------------------------------------------------------------
# Reuse of an already-crossed container
# --------------------------------------------------------------------------

def _mock_docker_with_existing(monkeypatch, existing_mounts):
    """Mock a daemon that already has a reusable container with *existing_mounts*."""
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, list) else cmd)
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "ps":
            # Egress off widens the probe to ID\tState\tEgressLabel.
            return subprocess.CompletedProcess(
                cmd, 0, stdout="stale123\trunning\toff\n", stderr=""
            )
        if cmd[1] == "inspect" and "{{range .Mounts}}" in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 0, stdout="".join(f"{m}\n" for m in existing_mounts), stderr=""
            )
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="fresh-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _was_removed(calls, container_id="stale123"):
    return any(
        isinstance(c, list) and c[1:4] == ["rm", "-f", container_id] for c in calls
    )


def test_reused_container_with_foreign_mount_is_replaced(monkeypatch, two_profiles, caplog):
    """A persistent container already holding Alice's dirs cannot serve Bob."""
    foreign = two_profiles["alice"] / "cache" / "documents"
    calls = _mock_docker_with_existing(monkeypatch, [str(foreign)])

    with caplog.at_level("ERROR"):
        env = _make_env(persist_across_processes=True)

    assert _was_removed(calls), "the crossed container should have been removed"
    assert env._container_id == "fresh-container-id"
    assert "bind-mounts profile 'alice'" in caplog.text


def test_reused_container_with_own_mounts_is_kept(monkeypatch, two_profiles):
    """A clean container is still reused — no churn on every startup."""
    own = two_profiles["bob"] / "cache" / "documents"
    calls = _mock_docker_with_existing(monkeypatch, [str(own)])

    env = _make_env(persist_across_processes=True)

    assert not _was_removed(calls)
    assert env._container_id == "stale123"


def test_reuse_inspect_failure_keeps_legacy_behaviour(monkeypatch, two_profiles):
    """An inspect that fails must not churn a working container."""
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, list) else cmd)
        if not isinstance(cmd, list) or len(cmd) < 2:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="stale123\trunning\toff\n", stderr=""
            )
        if cmd[1] == "inspect" and "{{range .Mounts}}" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="daemon busy")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="fresh-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    env = _make_env(persist_across_processes=True)

    assert not _was_removed(calls)
    assert env._container_id == "stale123"


def test_reuse_boundary_check_skipped_for_shared_key(monkeypatch, two_profiles):
    """Opt-in sharing keeps its container even with sibling-profile mounts."""
    foreign = two_profiles["alice"] / "cache" / "documents"
    calls = _mock_docker_with_existing(monkeypatch, [str(foreign)])

    env = _make_env(persist_across_processes=True, shared_container_key="family")

    assert not _was_removed(calls)
    assert env._container_id == "stale123"
