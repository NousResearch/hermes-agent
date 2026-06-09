import pytest

from tools.terminal_tool import _get_env_config


def test_local_backend_ignores_malformed_docker_json_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "not json")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not json")
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", "not json")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", "not json")

    config = _get_env_config()

    assert config["env_type"] == "local"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}
    assert config["docker_extra_args"] == []
    assert config["docker_forward_env"] == []


def test_docker_backend_still_rejects_malformed_docker_json_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "not json")

    with pytest.raises(ValueError, match="TERMINAL_DOCKER_VOLUMES"):
        _get_env_config()
