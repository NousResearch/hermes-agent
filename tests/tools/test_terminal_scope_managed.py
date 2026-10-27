"""Managed terminal settings reach the same real consumers as a profile's own settings."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("user_yaml", [None, "terminal:\n  backend: local\n  timeout: 37\n"])
def test_managed_terminal_policy_reaches_routed_consumers(tmp_path, monkeypatch, user_yaml):
    import gateway.run as gw
    from gateway.platforms.base import _parse_docker_volume_mounts
    from tools.terminal_tool import _get_env_config
    from tools.terminal_scope import get_terminal_scope, terminal_env

    home = tmp_path / "profile"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "another-profile")
    (home / ".env").write_text("TERMINAL_ENV=local\n", encoding="utf-8")
    if user_yaml is not None:
        (home / "config.yaml").write_text(user_yaml, encoding="utf-8")
    policy = managed / "config.yaml"
    policy.write_text(
        "terminal:\n  backend: docker\n  docker_image: managed:test\n"
        "  docker_volumes: [/tmp/managed-work:/work:ro]\n",
        encoding="utf-8",
    )
    with gw._profile_runtime_scope(home):
        cfg = _get_env_config()
        assert cfg["env_type"] == "docker"
        assert cfg["docker_image"] == "managed:test"
        assert cfg["docker_volumes"] == ["/tmp/managed-work:/work:ro"]
        assert _parse_docker_volume_mounts() == [(Path("/tmp/managed-work").resolve(), Path("/work"))]
        assert cfg["docker_shared_container_key"] == ""
        assert terminal_env("TERMINAL_DOCKER_IMAGE_PINNED") == "1"
        if user_yaml is not None:
            assert terminal_env("TERMINAL_TIMEOUT") == "37"

    # A new scope must observe an administrator edit without mutating the launch process.
    policy.write_text("terminal:\n  backend: ssh\n  ssh_host: managed-host\n", encoding="utf-8")
    with gw._profile_runtime_scope(home):
        cfg = _get_env_config()
        assert (cfg["env_type"], cfg["ssh_host"]) == ("ssh", "managed-host")
        assert json.loads(terminal_env("TERMINAL_DOCKER_VOLUMES", "[]")) == []
    assert get_terminal_scope() is None
    assert os.environ["TERMINAL_DOCKER_SHARED_CONTAINER_KEY"] == "another-profile"


def test_broken_user_config_still_refuses_after_effective_loader_recovery(tmp_path, monkeypatch):
    import gateway.run as gw
    from hermes_cli.config_effective import load_user_config_effective
    from tools.terminal_scope import TerminalPolicyUnavailable, terminal_env

    home = tmp_path / "profile"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")
    config = home / "config.yaml"
    config.write_text("terminal:\n  timeout: 37\n", encoding="utf-8")
    load_user_config_effective(config)
    config.write_text("terminal: [unterminated", encoding="utf-8")
    load_user_config_effective(config)  # Other readers may have served last-known-good state.
    with gw._profile_runtime_scope(home):
        with pytest.raises(TerminalPolicyUnavailable):
            terminal_env("TERMINAL_ENV")
