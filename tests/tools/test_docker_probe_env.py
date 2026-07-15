"""Docker _run_bash env-forwarding regression for #56634.

The ambient-PATH probe and the login bootstrap both run before the snapshot
exists, so both must receive the same ``-e`` init env -- otherwise the probe
captures the image-default PATH instead of a configured one. Injection keys on
``not _snapshot_ready`` rather than ``login``.
"""
from __future__ import annotations

import pytest

docker = pytest.importorskip("tools.environments.docker")


def _stub_env(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        docker, "_popen_bash", lambda cmd, stdin_data=None: captured.setdefault("cmd", cmd)
    )
    env = docker.DockerEnvironment.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    env._container_id = "cid"
    env._init_env_args = ["-e", "FOO=bar"]
    return env, captured


def test_probe_gets_init_env_before_snapshot_ready(monkeypatch):
    env, captured = _stub_env(monkeypatch)
    env._snapshot_ready = False
    env._run_bash("builtin printf x", login=False)  # the ambient probe
    assert "-e" in captured["cmd"] and "FOO=bar" in captured["cmd"]


def test_commands_omit_init_env_after_snapshot_ready(monkeypatch):
    env, captured = _stub_env(monkeypatch)
    env._snapshot_ready = True
    env._run_bash("echo hi", login=False)
    assert "FOO=bar" not in captured["cmd"]
