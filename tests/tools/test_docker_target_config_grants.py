"""Target ``terminal.docker_env`` grants must survive source-residue removal.

When launch profile A's ``.env`` owns a name and routed profile B declares that
name only in B's ``terminal.docker_env``, the Docker client child must receive
B's configured value (paired with a name-only ``-e NAME`` flag), never A's value
and never an empty slot. Ambient and force-prefixed extras gain nothing.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from agent.secret_scope import build_profile_env_boundary
from tools.environments import base_output, docker


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    for home in (source, target):
        home.mkdir()
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (source / ".env").write_text(
        "DB_DSN=a-dsn\nDB_PASSWORD=a-password\n", encoding="utf-8")
    (target / ".env").write_text("TARGET_DOTENV=b-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("DB_DSN", "a-dsn")
    monkeypatch.setenv("DB_PASSWORD", "a-password")
    monkeypatch.setenv("_HERMES_FORCE_SMUGGLED", "ambient-force")
    monkeypatch.setenv("GH_TOKEN", "ambient-provider")
    from tools.env_passthrough import clear_env_passthrough
    clear_env_passthrough()
    return source, target


def _docker_env(boundary, config_env):
    env = docker.DockerEnvironment.__new__(docker.DockerEnvironment)
    env._container_id, env._docker_exe = "container", "docker"
    env._forward_env = []
    env._env = dict(config_env)
    env._profile_env_boundary = boundary
    env._profile_scoped_passthrough = True
    return env


@pytest.mark.parametrize("login", [True, False], ids=["init", "runtime"])
def test_target_docker_env_grants_survive_source_owned_collisions(profiles, monkeypatch, login):
    source, target = profiles
    config_env = {"DB_DSN": "b-dsn", "DB_PASSWORD": "b-password", "B_ONLY": "b-only"}
    boundary = build_profile_env_boundary(source, target)
    env = _docker_env(boundary, {**config_env, "BWS_ACCESS_TOKEN": "internal-secret"})
    captured = []
    monkeypatch.setattr(base_output.subprocess, "Popen",
                        lambda cmd, **kw: captured.append((cmd, kw)) or Mock())

    if login:
        env._init_env_args = env._build_init_env_args()
    else:
        # Runtime commands forward only passthrough grants; docker_env reached the
        # container through init, so the runtime child must still drop A's values.
        env._init_env_args = []
    env._run_bash("true", login=login)

    cmd, kwargs = captured[0]
    child = kwargs["env"]
    argv = " ".join(cmd)
    for name, value in config_env.items():
        if login:
            assert child.get(name) == value, name
            assert f"-e {name}" in argv
        else:
            assert child.get(name) not in ("a-dsn", "a-password"), name
        assert value not in argv  # values never ride the world-readable argv
    assert "a-dsn" not in child.values() and "a-password" not in child.values()
    assert "SMUGGLED" not in child and "_HERMES_FORCE_SMUGGLED" not in child
    assert "BWS_ACCESS_TOKEN" not in child  # config grants never carry Hermes-internal secrets
    assert "GH_TOKEN" not in child
    assert child["HERMES_HOME"] == str(target)
