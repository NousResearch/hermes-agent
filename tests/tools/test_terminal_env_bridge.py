"""Behavioral regressions for the terminal config → env snapshot.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables through a
bridged snapshot (``_terminal_env_snapshot()``) rather than process-global
env.  Explicit terminal keys in config.yaml override stale launcher/.env
values; environment values for keys omitted from config.yaml are preserved;
process-global env is never mutated.
"""

import os

import pytest

import tools.terminal_tool as terminal_tool
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with clean TERMINAL_* env."""
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_CWD",
        "TERMINAL_DOCKER_IMAGE",
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
    # Snapshot semantics: process-global env is not mutated.
    assert "TERMINAL_ENV" not in os.environ


def test_explicit_config_backend_overrides_stale_env(monkeypatch):
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert os.environ["TERMINAL_ENV"] == "local"


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

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "~"
    assert "TERMINAL_CWD" not in os.environ


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
    assert "TERMINAL_ENV" not in os.environ


def test_snapshot_is_fresh_per_call(monkeypatch):
    """Each ``_get_env_config()`` bridges a fresh snapshot; no global state.

    The old process-global ``_ensure_terminal_env_bridged()`` ran at most once;
    the snapshot design has no such optimization, so repeated calls stay
    consistent without carrying state between them.
    """
    calls = []

    import hermes_cli.config as config_mod

    real = config_mod.apply_terminal_config_to_env

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", _counting)
    _write_config("{}\n")

    first = terminal_tool._get_env_config()
    second = terminal_tool._get_env_config()

    assert len(calls) == 2
    assert first == second


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

    # Fail-closed: TERMINAL_ENV (the backend selection) survives the bridge
    # failure, but the remaining TERMINAL_* settings are stripped so stale
    # values are never trusted.
    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == ""


def test_worker_timeout_override_survives_bridge(monkeypatch):
    """Explicit worker-scoped overrides beat config defaults.

    Subprocess callers (e.g. kanban ``_default_spawn``) set
    ``TERMINAL_TIMEOUT``/``TERMINAL_LIFETIME_SECONDS`` deliberately; the
    snapshot must restore them after the bridge applies config defaults.
    """
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  timeout: 180\n"
        "  lifetime_seconds: 300\n"
    )
    monkeypatch.setenv("TERMINAL_TIMEOUT", "600")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "900")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["timeout"] == 600
    assert config["lifetime_seconds"] == 900


def test_bridge_applies_config_default_when_no_worker_override(monkeypatch):
    """Without an explicit worker override, config defaults apply."""
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  timeout: 240\n"
    )
    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)

    config = terminal_tool._get_env_config()

    assert config["timeout"] == 240
