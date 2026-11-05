"""Snap-compatible runtime-label staging for docker run (#84969)."""

import subprocess
from pathlib import Path

import pytest

from tools.environments import docker as docker_env


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        forward_env=kwargs.get("forward_env"),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
        extra_args=kwargs.get("extra_args", []),
        persist_across_processes=kwargs.get("persist_across_processes", True),
        shared_container_key=kwargs.get("shared_container_key", ""),
        shm_size=kwargs.get("shm_size", docker_env._DEFAULT_SHM_SIZE),
        snap_compat=kwargs.get("snap_compat", False),
        image_pinned=kwargs.get("image_pinned", False),
    )


def test_snap_compat_stages_label_file_under_home(monkeypatch, tmp_path):
    """Snap-confined Docker cannot read host /tmp: with snap_compat the runtime
    label file is staged in a random mode-0700 non-hidden directory under $HOME
    and removed after both success and exit-125 runs."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)

    seen = {}

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run" and "--label-file" in cmd:
            label_path = Path(cmd[cmd.index("--label-file") + 1])
            seen["path"] = label_path
            seen["exists_during_run"] = label_path.is_file()
            seen["mode"] = label_path.parent.stat().st_mode & 0o777
        return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _make_dummy_env(snap_compat=True)

    label_path = seen["path"]
    assert seen["exists_during_run"] is True
    assert label_path.parent.parent == home
    assert not label_path.parent.name.startswith(".")
    assert seen["mode"] == 0o700
    assert not label_path.exists() and not label_path.parent.exists()

    def _fail(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run" and "--label-file" in cmd:
            seen["125_path"] = Path(cmd[cmd.index("--label-file") + 1])
            raise subprocess.CalledProcessError(125, cmd, output="", stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _fail)
    with pytest.raises(subprocess.CalledProcessError):
        _make_dummy_env(snap_compat=True)
    assert seen["125_path"].parent.parent == home
    assert not seen["125_path"].exists() and not seen["125_path"].parent.exists()


def test_default_label_file_stays_in_system_temp(monkeypatch, tmp_path):
    """Without snap_compat the label staging keeps using the system temp dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)

    seen = {}

    def _run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run" and "--label-file" in cmd:
            seen["path"] = Path(cmd[cmd.index("--label-file") + 1])
        return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    _make_dummy_env()

    assert home not in seen["path"].parents
    assert not seen["path"].exists()
