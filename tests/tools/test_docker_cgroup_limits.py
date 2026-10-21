"""Tests for cgroup resource-limit gating in the docker backend.

On hosts where the cgroup v2 cpu/memory/pids controllers are not delegated
(e.g. unprivileged Proxmox LXCs), passing ``--cpus``/``--memory``/``--pids-limit``
to ``docker run`` fails every container start with OCI runtime error / exit 126.
``_cgroup_limits_available`` probes once and the resource flags are gated on it,
so the sandbox degrades gracefully instead of failing.
"""
import subprocess
from itertools import pairwise

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


# What DockerEnvironment hands the probe for a root sandbox: that sandbox's own security args.
_ROOT_POSTURE = docker_env._build_security_args(False)


def test_probe_returns_true_when_container_starts(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("hermes-agent:latest", _ROOT_POSTURE) is True
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
    docker_env._cgroup_limits_available("img", _ROOT_POSTURE)
    docker_env._cgroup_limits_available("img", _ROOT_POSTURE)
    docker_env._cgroup_limits_available("img", _ROOT_POSTURE)
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
    assert docker_env._cgroup_limits_available("img", _ROOT_POSTURE) is False
    assert docker_env._cgroup_limits_available("img", _ROOT_POSTURE) is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    "info, create, cached",
    [
        # `docker info` failing (daemon cold-start) is transient: not "not overlay2".
        ((1, "", "Cannot connect to the Docker daemon"), None, None),
        # The daemon rejecting --storage-opt is a host property: cached False.
        ((0, "overlay2\n", ""), (125, "", "Error response from daemon: --storage-opt is "
                                          "supported only for overlay2 with pquota"), False),
        # A create failure that never reached the storage check (pull error) is transient.
        ((0, "overlay2\n", ""), (125, "", "Unable to find image 'hello-world:latest'"), None),
    ],
)
def test_storage_opt_probe_caches_only_definitive_answers(monkeypatch, info, create, cached):
    """``_storage_opt_supported`` must not latch disk quota off for the process on a
    failure that says nothing about pquota support (#116162)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    def _run(cmd, *a, **k):
        rc, out, err = info if cmd[1] == "info" else create
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env.DockerEnvironment._storage_opt_supported() is False
    assert docker_env._storage_opt_ok is cached


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


# ── Probe hardening ───────────────────────────────────────────────
# The probe is a container from the sandbox image that comes up BEFORE the sandbox. It
# must not be the one container that escapes the sandbox's hardening.


def _pairs(argv):
    """``(flag, value)`` pairs of a ``docker run`` argv, ``--flag=value`` read as ``--flag value``."""
    tokens = [t for a in argv for t in (a.split("=", 1) if a.startswith("--") else (a,))]
    return set(pairwise(tokens))


def _posture(argv):
    """The cap / no-new-privileges / user pairs of a ``docker run`` argv."""
    return {(f, v) for f, v in _pairs(argv) if f in ("--cap-drop", "--cap-add", "--security-opt", "--user")}


def _tmpfs_specs(argv):
    return {v for f, v in _pairs(argv) if f == "--tmpfs"}


# docker_extra_args carrying a host mount and a host env name (``--flag=value`` spelling).
_EXTRA_ARGS = ["--volume=/srv/extra:/extra", "--env=EXTRA_TOKEN"]
# Flags that hand a container something of the host: a mount or an env value.
_MOUNT_ENV_FLAGS = ("-v", "--volume", "--mount", "--volumes-from", "-e", "--env", "--env-file")


@pytest.mark.parametrize("mode", [{}, {"run_as_host_user": True}, {"snap_compat": True}],
                         ids=["root", "host-user", "snap-compat"])
def test_probe_carries_the_sandbox_posture_and_nothing_of_the_host(monkeypatch, mode):
    """E2e through DockerEnvironment, per privilege mode, for a persistent sandbox (the one with
    the most host mounts). The probe carries exactly the cap / no-new-privileges / ``--user`` pairs
    of the sandbox's ``run -d`` and its nosuid/noexec tmpfs for /tmp and /var/tmp. On top, it is
    air-gapped although the sandbox is networked, read-only, runs from ``/`` with the image
    entrypoint cleared (``sleep 0`` needs none of it), and none of the sandbox's mounts, env or
    docker_extra_args reach it."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(docker_env.os, "getgid", lambda: 1000, raising=False)
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)  # no docker exec
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "run" and "-d" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env.DockerEnvironment(image="img", task_id="t", volumes=["/srv/data:/data"], env={"TOKEN": "x"},
                                 extra_args=_EXTRA_ARGS, persistent_filesystem=True, **mode)

    runs = [c for c in calls if isinstance(c, list) and len(c) > 1 and c[1] == "run"]
    probe = next(c for c in runs if c[-2:] == ["sleep", "0"])
    sandbox = next(c for c in runs if "-d" in c)
    assert ("--cap-drop", "ALL") in _posture(probe)  # so the equality below is not vacuous
    assert _posture(probe) == _posture(sandbox)
    assert {s for s in _tmpfs_specs(sandbox) if s.startswith(("/tmp:", "/var/tmp:"))} <= _tmpfs_specs(probe)
    assert docker_env._extra_args_network_mode(probe) == "none"
    assert docker_env._extra_args_network_mode(sandbox) != "none"
    assert "--read-only" in probe and {("-w", "/"), ("--entrypoint", "")} <= _pairs(probe)
    assert probe[-3:] == ["img", "sleep", "0"]  # every flag precedes the image
    assert any(t in _MOUNT_ENV_FLAGS for t in sandbox)  # sandbox, skills/cache and user mounts, env
    assert not [t for t in probe if t.split("=", 1)[0] in _MOUNT_ENV_FLAGS]
    host_data = ("/srv/", "TOKEN")  # what the fixture mounts and forwards
    assert all(any(h in t for t in sandbox) for h in host_data) and set(_EXTRA_ARGS) <= set(sandbox)
    assert not [t for t in probe if any(h in t for h in host_data)]
