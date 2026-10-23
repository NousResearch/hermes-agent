"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os
from pathlib import Path

import pytest

from agent import runtime_cwd
from tools import terminal_tool
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an un-attempted bridge and clean mapped env."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    monkeypatch.setattr(runtime_cwd, "_cli_owns_cwd", False)
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


def test_local_cli_keeps_its_launch_cwd_through_config_bridges(tmp_path, monkeypatch):
    """#86411: a local CLI runs where it was launched, even with an explicit terminal.cwd.

    Every ``hermes`` process bridges config at import, before it knows it is a CLI, again on each
    dotenv reload, and once more on the terminal tool's first use. ``hermes chat`` and ``-z`` claim
    the launch dir (``--in`` or a resumed session already moved it), so none of these re-pins it.
    """
    from hermes_cli.env_loader import load_hermes_dotenv

    configured = tmp_path / "configured"
    launch = tmp_path / "launch"
    configured.mkdir()
    launch.mkdir()
    _write_config(f"terminal:\n  backend: local\n  cwd: '{configured}'\n")
    monkeypatch.chdir(launch)
    load_hermes_dotenv(load_external_secrets=False)

    runtime_cwd.claim_local_cli_cwd()
    load_hermes_dotenv(load_external_secrets=False)

    assert terminal_tool._get_env_config()["cwd"] == os.getcwd()
    assert runtime_cwd.resolve_agent_cwd() == Path(os.getcwd())


@pytest.mark.parametrize(
    ("backend", "claimed"), [("local", False), ("ssh", True)], ids=["not-a-local-cli", "remote-backend"])
def test_terminal_cwd_keeps_config_authority_outside_a_local_cli(tmp_path, monkeypatch, backend, claimed):
    """Gateways, cron, ACP and the dashboard never claim a cwd, even after importing cli.py (which
    exports its launch dir); a remote backend's cwd is never a host launch dir."""
    from hermes_cli.env_loader import load_hermes_dotenv

    local_dir = tmp_path / "configured"
    local_dir.mkdir()
    configured = "~/work" if backend == "ssh" else str(local_dir)
    _write_config(f"terminal:\n  backend: {backend}\n  cwd: '{configured}'\n")
    load_hermes_dotenv(load_external_secrets=False)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    if claimed:
        runtime_cwd.claim_local_cli_cwd()

    assert terminal_tool._get_env_config()["cwd"] == configured


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


def test_secondary_home_override_does_not_latch_ambient_env(tmp_path, monkeypatch):
    """#107422: first bridge under a secondary profile must not poison os.environ.

    Multiplexed dashboard sets ``set_hermes_home_override`` for profile B. If
    ``_ensure_terminal_env_bridged`` ran there (no terminal scope yet), the
    one-shot latch used to write B's docker policy into process-global env and
    every later unscoped launch-profile tool call inherited it.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch_home = tmp_path / "launch"
    secondary_home = tmp_path / "profiles" / "docker-bee"
    launch_home.mkdir(parents=True)
    secondary_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )
    (secondary_home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: bee/local:1\n"
        '  docker_volumes:\n'
        '    - /bee/vol:/data\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    # Clean ambient — the dashboard process starts without TERMINAL_ENV.
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_DOCKER_VOLUMES",
    ):
        monkeypatch.delenv(name, raising=False)

    token = set_hermes_home_override(str(secondary_home))
    try:
        # Unscoped call under secondary home (the residual path).
        terminal_tool._ensure_terminal_env_bridged()
    finally:
        reset_hermes_home_override(token)

    assert "TERMINAL_ENV" not in os.environ
    assert "TERMINAL_DOCKER_IMAGE" not in os.environ
    assert "TERMINAL_DOCKER_VOLUMES" not in os.environ
    # Bridge must still be available for the real launch profile afterwards.
    assert terminal_tool._terminal_config_bridge_attempted is False

    config = terminal_tool._get_env_config()
    assert config["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"
    assert "bee/local:1" not in os.environ.get("TERMINAL_DOCKER_IMAGE", "")
