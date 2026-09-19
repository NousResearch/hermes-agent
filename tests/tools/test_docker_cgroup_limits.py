"""Tests for cgroup resource-limit gating in the docker backend.

On hosts where the cgroup v2 cpu/memory/pids controllers are not delegated
(e.g. unprivileged Proxmox LXCs), passing ``--cpus``/``--memory``/``--pids-limit``
to ``docker run`` fails every container start with OCI runtime error / exit 126.
``_cgroup_limits_available`` probes once and the resource flags are gated on it,
so the sandbox degrades gracefully instead of failing.
"""
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


def test_pids_limit_not_in_base_security_args():
    """``--pids-limit`` must NOT be hardcoded in the static security args.

    It requires the pids cgroup controller and is gated on the probe instead.
    """
    assert "--pids-limit" not in docker_env._BASE_SECURITY_ARGS


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


def test_probe_timeout_is_not_cached_and_retried(monkeypatch):
    """A transient probe failure (auto-pull of an uncached image exceeding the
    60s timeout, daemon cold-start) must not disable resource limits for the
    process lifetime. The spawn degrades but the next spawn re-probes."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 60)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("img") is False
    assert docker_env._cgroup_limits_available("img") is True
    assert len(calls) == 2


def test_probe_non_cgroup_failure_is_not_cached(monkeypatch):
    """A nonzero probe exit for reasons unrelated to cgroups (manifest/pull
    error, daemon error) is not a definitive negative, so it is retried on the
    next spawn."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                cmd, 125, stdout="", stderr="docker: Error response from daemon: "
                "manifest for hermes-agent:latest not found: manifest unknown")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("img") is False
    assert docker_env._cgroup_limits_available("img") is True
    assert len(calls) == 2


def test_probe_pull_error_naming_cgroup_is_not_cached(monkeypatch):
    """An image whose NAME contains 'cgroup' produces a pull error mentioning it;
    that must not be read as a definitive cgroup rejection."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                cmd, 125, stdout="", stderr="docker: Error response from daemon: "
                "pull access denied for cgroup-tools, repository does not exist")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("cgroup-tools") is False
    assert docker_env._cgroup_limits_available("cgroup-tools") is True
    assert len(calls) == 2


def test_probe_definitive_cgroup_failure_is_cached(monkeypatch):
    """A daemon rejection that names cgroups IS a host property; caching it is
    the point of the probe, so subsequent spawns do not pay it again."""
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


def test_missing_docker_is_not_cached(monkeypatch):
    """find_docker() missing says nothing about cgroup support, so it is not cached."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    assert docker_env._cgroup_limits_available("img") is False

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run",
                        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    assert docker_env._cgroup_limits_available("img") is True


class TestStorageOptProbeCaching:
    """``_storage_opt_supported`` has the same transient-vs-definitive contract:
    only a real answer (driver name, or a daemon rejection naming storage) may
    be cached process-wide; timeouts and pull/daemon failures must retry."""

    def _fake_run(self, calls, info=None, create=None, info_exc=None):
        def _run(cmd, *a, **k):
            calls.append(cmd)
            sub = cmd[1] if isinstance(cmd, list) and len(cmd) > 1 else ""
            if sub == "info":
                if info_exc:
                    raise info_exc
                rc, stdout, stderr = info
                return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)
            if sub == "create":
                rc, stdout, stderr = create
                return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return _run

    def test_info_failure_not_cached(self, monkeypatch):
        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
        calls = []
        monkeypatch.setattr(docker_env.subprocess, "run", self._fake_run(
            calls, info=(1, "", "Cannot connect to the Docker daemon")))

        assert docker_env.DockerEnvironment._storage_opt_supported() is False
        monkeypatch.setattr(docker_env.subprocess, "run", self._fake_run(
            calls, info=(0, "overlay2\n", ""), create=(0, "cid123\n", "")))
        assert docker_env.DockerEnvironment._storage_opt_supported() is True

    def test_info_exception_not_cached(self, monkeypatch):
        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
        calls = []
        monkeypatch.setattr(docker_env.subprocess, "run", self._fake_run(
            calls, info_exc=subprocess.TimeoutExpired("docker", 10)))
        assert docker_env.DockerEnvironment._storage_opt_supported() is False
        assert docker_env._storage_opt_ok is None

    def test_unsupported_create_error_cached(self, monkeypatch):
        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
        calls = []
        monkeypatch.setattr(docker_env.subprocess, "run", self._fake_run(
            calls, info=(0, "overlay2\n", ""),
            create=(125, "", "Error response from daemon: --storage-opt is "
                             "supported only for overlay2 with pquota")))
        assert docker_env.DockerEnvironment._storage_opt_supported() is False
        assert docker_env.DockerEnvironment._storage_opt_supported() is False
        assert len(calls) == 2  # info + create once; cached thereafter

    def test_create_pull_failure_not_cached(self, monkeypatch):
        """A create-probe failure that never reached the storage check (image
        pull error) is transient; retried, not latched."""
        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
        calls = []
        monkeypatch.setattr(docker_env.subprocess, "run", self._fake_run(
            calls, info=(0, "overlay2\n", ""),
            create=(125, "", "Unable to find image 'hello-world:latest'")))
        assert docker_env.DockerEnvironment._storage_opt_supported() is False
        assert docker_env._storage_opt_ok is None


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
