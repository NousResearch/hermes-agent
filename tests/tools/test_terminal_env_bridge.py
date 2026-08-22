"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os
import time
from unittest.mock import patch

import pytest

import tools.terminal_tool as terminal_tool
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from agent.secret_scope import reset_secret_scope, set_secret_scope


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an un-attempted bridge and clean mapped env."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_CWD",
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_CONTAINER_PERSISTENT",
        "TERMINAL_LIFETIME_SECONDS",
        "TERMINAL_DEGRADED_MODE",
        "TERMINAL_SSH_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _write_config(text: str) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text)


def test_unset_terminal_env_backfills_backend_from_config():
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: custom/image:1\n"
    )

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "custom/image:1"
    assert os.environ["TERMINAL_ENV"] == "docker"


def test_explicit_config_backend_overrides_stale_env(monkeypatch):
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert os.environ["TERMINAL_ENV"] == "docker"


def test_partial_terminal_config_preserves_unrelated_env_values(monkeypatch):
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "env/image:2"
    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "env/image:2"


def test_explicit_config_key_overrides_matching_env_value(monkeypatch):
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: config/image:1\n"
    )
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "config/image:1"


def test_ssh_config_preserves_remote_tilde_cwd(monkeypatch):
    """SSH ``~`` belongs to the remote user, not the Hermes host/container."""
    _write_config("terminal:\n  backend: ssh\n  cwd: '~'\n")
    monkeypatch.setenv("HOME", "/opt/data/home")
    monkeypatch.setenv("USERPROFILE", r"C:\opt\data\home")

    config = terminal_tool._get_env_config()

    assert os.environ["TERMINAL_CWD"] == "~"
    assert config["cwd"] == "~"


def test_env_is_preserved_when_config_has_no_terminal_section(monkeypatch):
    _write_config("agent:\n  max_turns: 100\n")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == "example.test"


def test_defaults_backfill_when_neither_config_nor_env_selects_backend():
    _write_config("{}\n")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"


def test_global_bridge_once_but_profile_overlay_runs_per_config_read(monkeypatch):
    calls = []

    import hermes_cli.config as config_mod

    real = config_mod.apply_terminal_config_to_env

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", _counting)
    _write_config("{}\n")

    terminal_tool._get_env_config()
    terminal_tool._get_env_config()

    # One call is the global one-shot bridge; each config read applies a
    # private profile-scoped overlay so idle recreation cannot use launch env.
    assert len(calls) == 3


def test_bridge_config_failure_does_not_crash(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: (_ for _ in ()).throw(RuntimeError("config read failed")),
    )
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == "example.test"


def test_routed_ssh_profile_recreates_ssh_after_idle_cleanup(monkeypatch, tmp_path):
    """A routed profile must not inherit the gateway's local process backend."""
    profile_home = tmp_path / "profiles" / "vps"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: ssh\n"
        "  ssh_host: remote.example.test\n"
        "  ssh_user: deploy\n"
        "  lifetime_seconds: 1\n"
    )
    monkeypatch.setenv("TERMINAL_ENV", "local")
    task_id = "routed-ssh"
    effective_task_id = terminal_tool._resolve_container_task_id(task_id)
    terminal_tool._active_environments.pop(effective_task_id, None)
    terminal_tool._last_activity.pop(effective_task_id, None)
    token = set_hermes_home_override(str(profile_home))
    created = []

    class FakeEnvironment:
        def cleanup(self):
            pass

    def fake_create(**kwargs):
        created.append(kwargs)
        return FakeEnvironment()

    try:
        with patch.object(terminal_tool, "_start_cleanup_thread"), patch.object(
            terminal_tool, "_create_environment", side_effect=fake_create
        ):
            assert terminal_tool.ensure_task_env(task_id) is not None
            terminal_tool._last_activity[effective_task_id] = time.time() - 2
            terminal_tool._cleanup_inactive_envs(lifetime_seconds=1)
            assert terminal_tool.ensure_task_env(task_id) is not None
    finally:
        reset_hermes_home_override(token)
        terminal_tool._active_environments.pop(effective_task_id, None)
        terminal_tool._last_activity.pop(effective_task_id, None)

    assert [call["env_type"] for call in created] == ["ssh", "ssh"]
    assert all(call["ssh_config"]["host"] == "remote.example.test" for call in created)


def test_routed_ssh_profile_missing_connection_config_fails_closed(monkeypatch, tmp_path):
    """An SSH-selected profile must raise, never fall through to local."""
    profile_home = tmp_path / "profiles" / "broken-vps"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("terminal:\n  backend: ssh\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    scope = set_secret_scope({})
    token = set_hermes_home_override(str(profile_home))
    try:
        config = terminal_tool._get_env_config()
        assert config["env_type"] == "ssh"
        with pytest.raises(ValueError, match="SSH environment requires"):
            terminal_tool._create_environment(
                env_type=config["env_type"], image="", cwd=config["cwd"],
                timeout=config["timeout"],
                ssh_config=terminal_tool._ssh_config_from_config(config),
            )
    finally:
        reset_hermes_home_override(token)
        reset_secret_scope(scope)


def test_routed_ssh_profile_does_not_inherit_gateway_ssh_settings(monkeypatch, tmp_path):
    """Missing profile SSH credentials cannot connect to the gateway's host."""
    profile_home = tmp_path / "profiles" / "isolated-vps"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("terminal:\n  backend: ssh\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "gateway.example.test")
    monkeypatch.setenv("TERMINAL_SSH_USER", "gateway-user")
    scope = set_secret_scope({})
    token = set_hermes_home_override(str(profile_home))
    try:
        config = terminal_tool._get_env_config()
    finally:
        reset_hermes_home_override(token)
        reset_secret_scope(scope)

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == ""
    assert config["ssh_user"] == ""


def test_routed_profile_direct_terminal_consumers_use_scoped_overlay(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "docker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: docker\n"
        "  container_persistent: false\n"
        "  lifetime_seconds: 30\n"
        "  degraded_mode: fail\n"
    )
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "300")
    monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "warn")
    monkeypatch.setattr(terminal_tool, "_docker_orphan_reaper_profiles", set())
    scope = set_secret_scope({})
    token = set_hermes_home_override(str(profile_home))
    try:
        with patch.object(terminal_tool.subprocess, "run") as sudo_probe:
            assert terminal_tool._sudo_nopasswd_works() is False
            sudo_probe.assert_not_called()
        assert terminal_tool._docker_session_isolation_enabled() is True
        assert terminal_tool._get_profile_terminal_env()["TERMINAL_DEGRADED_MODE"] == "fail"
        assert os.environ["TERMINAL_ENV"] == "local"
        with patch(
            "tools.environments.docker._get_active_profile_name",
            return_value="docker",
        ), patch("tools.environments.docker.reap_orphan_containers", return_value=[]) as reap:
            terminal_tool._maybe_reap_docker_orphans({"docker_orphan_reaper": True})
            assert reap.call_args.kwargs["max_age_seconds"] == 120
    finally:
        reset_hermes_home_override(token)
        reset_secret_scope(scope)
    assert os.environ["TERMINAL_ENV"] == "local"
