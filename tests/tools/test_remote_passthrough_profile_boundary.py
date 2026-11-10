"""Contracts at the shared Docker/SSH forwarding and subprocess boundary."""

from unittest.mock import Mock

import pytest

from agent.secret_scope import build_profile_env_boundary
from tools.environments import base_output, docker, ssh
from tools.environments.remote_common import resolve_passthrough_env


@pytest.mark.parametrize("backend", ["docker", "ssh"])
@pytest.mark.parametrize("value", ["target-value", "", None])
def test_remote_forwarding_uses_target_config_and_values(monkeypatch, tmp_path, backend, value):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(source))
    (source / "config.yaml").write_text(
        "terminal:\n  env_passthrough: [SOURCE_ONLY]\n", encoding="utf-8")
    (target / "config.yaml").write_text(
        "terminal:\n  env_passthrough: [TARGET_ONLY]\n", encoding="utf-8")
    (source / ".env").write_text("SOURCE_ONLY=source\nTARGET_ONLY=wrong\n", encoding="utf-8")
    (target / ".env").write_text(
        "" if value is None else f"TARGET_ONLY={value}\n", encoding="utf-8")
    monkeypatch.setenv("SOURCE_ONLY", "source")
    monkeypatch.setenv("TARGET_ONLY", "wrong")
    monkeypatch.setenv("GH_TOKEN", "ambient-provider")
    from tools.env_passthrough import clear_env_passthrough
    clear_env_passthrough()
    boundary = build_profile_env_boundary(source, target)
    captured = []
    monkeypatch.setattr(base_output.subprocess, "Popen", lambda cmd, **kw: captured.append((cmd, kw)) or Mock())
    if backend == "docker":
        env = docker.DockerEnvironment.__new__(docker.DockerEnvironment)
        env._container_id, env._docker_exe = "container", "docker"
        env._forward_env = []
    else:
        env = ssh.SSHEnvironment.__new__(ssh.SSHEnvironment)
        env.host, env.user, env.port, env.key_path = "example.invalid", "test", 22, ""
        env.control_socket = tmp_path / "control.sock"
    env._profile_env_boundary = boundary
    env._profile_scoped_passthrough = True
    env._run_bash("true")
    cmd, kwargs = captured[0]
    child = kwargs["env"]
    assert "SOURCE_ONLY" not in child
    assert "SOURCE_ONLY" not in " ".join(cmd)
    assert "GH_TOKEN" not in child
    assert child["HERMES_HOME"] == str(target)
    if value is None:
        assert "TARGET_ONLY" not in child
        assert "unset TARGET_ONLY" in " ".join(cmd)
    else:
        assert child["TARGET_ONLY"] == value
        assert "TARGET_ONLY" in " ".join(cmd)
        assert "target-value" not in " ".join(cmd)
    if backend == "ssh":
        assert env._control_socket_for(("TARGET_ONLY",)) != env._control_socket_for(())


@pytest.mark.parametrize("name", [
    "BWS_ACCESS_TOKEN", "auxiliary_x_api_key", "APPTAINERENV_OP_CONNECT_TOKEN",
    "SINGULARITYENV_APPTAINERENV_BWS_ACCESS_TOKEN",
])
def test_explicit_forwarding_excludes_internal_secrets_and_preserves_empty(monkeypatch, tmp_path, name):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(name, "internal-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    values, unsets = resolve_passthrough_env(
        [name, "OPENAI_API_KEY"],
        hermes_env_loader=lambda: {name: "internal-secret", "OPENAI_API_KEY": "stale"})
    assert values == {"OPENAI_API_KEY": ""}
    assert not unsets
