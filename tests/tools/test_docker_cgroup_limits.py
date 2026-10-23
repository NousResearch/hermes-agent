"""Tests for cgroup resource-limit gating in the docker backend.

On hosts where the cgroup v2 cpu/memory/pids controllers are not delegated
(e.g. unprivileged Proxmox LXCs), passing ``--cpus``/``--memory``/``--pids-limit``
to ``docker run`` fails every container start with OCI runtime error / exit 126.
``_cgroup_limits_available`` probes once and the resource flags are gated on it,
so the sandbox degrades gracefully instead of failing.
"""
import os
import subprocess

import pytest

import tools.environments.docker as docker_env


@pytest.fixture(autouse=True)
def _reset_cgroup_cache():
    """The probe results are cached in module-level globals; reset per test."""
    docker_env._cgroup_limits_ok = None
    docker_env._storage_opt_ok = None
    yield
    docker_env._cgroup_limits_ok = None
    docker_env._storage_opt_ok = None




def test_probe_returns_true_when_container_starts(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("hermes-agent:latest") is True
    # Probes all three controllers together against the real sandbox image.
    assert "--cpus" in captured["cmd"]
    assert "--memory" in captured["cmd"]
    assert "--pids-limit" in captured["cmd"]
    assert "hermes-agent:latest" in captured["cmd"]


def test_probe_result_is_cached(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env._cgroup_limits_available("img")
    docker_env._cgroup_limits_available("img")
    docker_env._cgroup_limits_available("img")
    assert len(calls) == 1  # probe runs once, then cached


def test_probe_definitive_cgroup_failure_is_cached(monkeypatch):
    """A daemon rejection that names cgroups IS a host property; caching it is
    the point of the probe, so subsequent spawns do not pay it again (#116162
    only stops caching failures that say nothing about cgroups)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 126, stdout="", stderr="docker: Error response from daemon: failed to "
            "create task for container: OCI runtime create failed: error setting cgroup "
            "config for procHooks process: permission denied")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("img") is False
    assert docker_env._cgroup_limits_available("img") is False
    assert len(calls) == 1


def _storage_opt_run(monkeypatch, *, driver="overlay2", backing="ext4",
                     root="/var/lib/docker", mounts=None, create=None):
    """Fake ``docker`` for the --storage-opt decision; returns the argv list it saw.

    ``docker info`` is rendered per requested ``--format`` the way the real daemon does, so
    the fake answers the old single-field format and the fingerprint format alike. Any
    ``docker create`` when *create* is None fails the test: on the hosts that can answer from
    ``docker info`` and the mount table there must be no throwaway container at all (#137493).
    """
    calls = []

    def _render_info(fmt):
        if fmt == "{{.Driver}}":
            return f"{driver}\n"
        if fmt == "{{.Driver}}|{{.DockerRootDir}}|{{json .DriverStatus}}":
            # `{{json .DriverStatus}}` marshals Go's [][]string: a list of [key, value] pairs.
            status = f'[["Backing Filesystem","{backing}"]]' if backing is not None else "[]"
            return f"{driver}|{root}|{status}\n"
        raise AssertionError(f"unsupported docker info format: {fmt}")

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "info":
            fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else "{{.Driver}}"
            return subprocess.CompletedProcess(cmd, 0, stdout=_render_info(fmt), stderr="")
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "rm":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if create is not None:
            rc, out, err = create
            return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)
        raise AssertionError(f"unexpected docker call: {cmd}")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "_proc_mounts", lambda: mounts)
    return calls


def _created(calls):
    return [c for c in calls if isinstance(c, list) and len(c) > 1 and c[1] == "create"]


@pytest.mark.parametrize(
    "info, create, cached",
    [
        # `docker info` failing (daemon cold-start) is transient: not "not overlay2".
        ((1, "", "Cannot connect to the Docker daemon"), None, None),
        # The daemon rejecting --storage-opt is a host property: cached False.
        ((0, "", ""), (125, "", "Error response from daemon: --storage-opt is "
                                "supported only for overlay over xfs with 'pquota' mount option"), False),
        # A create failure that never reached the storage check (pull error) is transient.
        ((0, "", ""), (125, "", "Unable to find image 'hello-world:latest'"), None),
    ],
)
def test_storage_opt_probe_caches_only_definitive_answers(monkeypatch, info, create, cached):
    """``_storage_opt_supported`` must not latch disk quota off for the process on a
    failure that says nothing about pquota support (#116162). An xfs-root daemon with no
    local mount covering it is the one case that still has to ask the daemon."""
    calls = _storage_opt_run(monkeypatch, backing="xfs", root="/var/lib/docker", create=create)
    monkeypatch.setattr(docker_env, "_proc_mounts", lambda: None)
    info_rc, info_out, info_err = info
    if info_rc:  # the daemon is unreachable: `docker info` itself fails
        def _run(cmd, *a, **k):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, info_rc, stdout=info_out, stderr=info_err)

        monkeypatch.setattr(docker_env.subprocess, "run", _run)

    assert docker_env.DockerEnvironment._storage_opt_supported() is False
    assert docker_env._storage_opt_ok is cached


def test_storage_opt_ext4_backing_is_decided_without_creating_a_container(monkeypatch):
    """The default Linux install — overlay2 on ext4 — rejects every ``docker create
    --storage-opt``, which is what spammed the daemon journal and left a dying container per
    worker process (#137493). The backing filesystem already answers, so nothing is created."""
    calls = _storage_opt_run(monkeypatch, backing="ext4")
    assert docker_env.DockerEnvironment._storage_opt_supported() is False
    assert docker_env._storage_opt_ok is False
    assert _created(calls) == []


def test_storage_opt_non_overlay2_driver_is_decided_without_creating_a_container(monkeypatch):
    """A driver that is not overlay2 answers on its own, with no probe."""
    calls = _storage_opt_run(monkeypatch, driver="vfs", backing="xfs")
    assert docker_env.DockerEnvironment._storage_opt_supported() is False
    assert docker_env._storage_opt_ok is False
    assert _created(calls) == []


@pytest.mark.parametrize("options, supported", [
    ("rw,pquota", True),
    ("rw,prjquota", True),  # what /proc/mounts actually spells for project quotas
    ("rw,relatime", False),
])
def test_storage_opt_xfs_mount_options_decide_without_creating_a_container(
        monkeypatch, tmp_path, options, supported):
    """overlay2 on xfs: the local mount options decide, so the daemon is never asked to
    create anything."""
    root = tmp_path / "docker"
    root.mkdir()
    mounts = f"/dev/sda1 {os.path.realpath(str(root))} xfs {options} 0 0\n"
    calls = _storage_opt_run(monkeypatch, backing="xfs", root=str(root), mounts=mounts)
    assert docker_env.DockerEnvironment._storage_opt_supported() is supported
    assert docker_env._storage_opt_ok is supported
    assert _created(calls) == []


def test_storage_opt_xfs_root_without_a_local_mount_still_asks_the_daemon(monkeypatch, tmp_path):
    """A daemon whose xfs root no local mount covers (remote/SSH, exotic bind layouts) is the
    only case left to the container probe, and its container is still removed."""
    root = tmp_path / "docker"
    root.mkdir()
    calls = _storage_opt_run(monkeypatch, backing="xfs", root=str(root),
                             mounts="/dev/sda1 / ext4 rw 0 0\n",
                             create=(0, "deadbeefcafe\n", ""))
    assert docker_env.DockerEnvironment._storage_opt_supported() is True
    assert docker_env._storage_opt_ok is True
    assert len(_created(calls)) == 1
    assert any(c[1] == "rm" and "deadbeefcafe" in c for c in calls)


def test_transient_probe_failure_recovers_on_next_spawn(monkeypatch):
    """E2e through DockerEnvironment: the probe timing out on spawn one (auto-pull
    past the 60s timeout, daemon cold-start) leaves THAT container unlimited but
    must not disable limits process-wide. Spawn two re-probes, succeeds, and its
    docker run argv carries --cpus/--memory/--pids-limit."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls, probe_calls = [], []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "run" and "--rm" in cmd:
            # the throwaway cgroup probe (`run --rm ... sleep 0`), not the real `run -d`
            probe_calls.append(cmd)
            if len(probe_calls) == 1:
                raise subprocess.TimeoutExpired(cmd, 60)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "run" and "-d" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env.DockerEnvironment(image="img", cpu=1.0, memory=512, task_id="t1")
    docker_env.DockerEnvironment(image="img", cpu=1.0, memory=512, task_id="t2")

    run_argvs = [c for c in calls
                 if isinstance(c, list) and len(c) > 1 and c[1] == "run" and "-d" in c]
    assert len(run_argvs) == 2
    assert "--cpus" not in run_argvs[0] and "--memory" not in run_argvs[0]
    assert "--cpus" in run_argvs[1] and "1.0" in run_argvs[1]
    assert "--memory" in run_argvs[1] and "512m" in run_argvs[1]
    assert "--pids-limit" in run_argvs[1]
    assert len(probe_calls) == 2  # re-probed, not latched off the timeout
