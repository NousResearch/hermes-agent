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
    """The probe result is cached in a module-level global; reset per test."""
    docker_env._cgroup_limits_ok = None
    yield
    docker_env._cgroup_limits_ok = None


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


def test_probe_returns_false_and_warns_on_oci_error(monkeypatch, caplog):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    def _run(cmd, *a, **k):
        return subprocess.CompletedProcess(
            cmd, 126, stdout="",
            stderr="crun: controller `pids` is not available",
        )

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    with caplog.at_level("WARNING"):
        assert docker_env._cgroup_limits_available("img") is False
    assert "Cgroup resource limits" in caplog.text


def test_probe_returns_false_when_no_docker(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    assert docker_env._cgroup_limits_available("img") is False


def test_probe_returns_false_on_empty_image(monkeypatch):
    """An empty image string must not be probed (would be a malformed run)."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        docker_env.subprocess, "run",
        lambda *a, **k: pytest.fail("should not probe with empty image"),
    )
    assert docker_env._cgroup_limits_available("") is False


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
