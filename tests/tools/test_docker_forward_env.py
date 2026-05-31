"""Regression tests for Docker env forwarding."""

from unittest.mock import patch

from tools.environments import docker as docker_mod


def _make_env(*, forward_env=None, env=None):
    instance = docker_mod.DockerEnvironment.__new__(docker_mod.DockerEnvironment)
    instance._forward_env = list(forward_env or [])
    instance._env = dict(env or {})
    return instance


def _env_value(args, key):
    prefix = f"{key}="
    for value in args:
        if value.startswith(prefix):
            return value.split("=", 1)[1]
    raise AssertionError(f"{key} was not forwarded: {args!r}")


def test_docker_forward_env_empty_live_value_falls_back_to_disk_env(monkeypatch):
    env = _make_env(forward_env=["LINEAR_API_KEY"])
    monkeypatch.setattr(
        docker_mod,
        "_load_hermes_env_vars",
        lambda: {"LINEAR_API_KEY": "x" * 48},
    )

    with patch.dict("os.environ", {"LINEAR_API_KEY": ""}, clear=False):
        args = env._build_init_env_args()

    assert _env_value(args, "LINEAR_API_KEY") == "x" * 48


def test_docker_forward_env_never_forwards_blank_secret(monkeypatch):
    env = _make_env(forward_env=["LINEAR_API_KEY"])
    monkeypatch.setattr(docker_mod, "_load_hermes_env_vars", lambda: {})

    with patch.dict("os.environ", {"LINEAR_API_KEY": ""}, clear=False):
        args = env._build_init_env_args()

    assert "LINEAR_API_KEY=" not in args
