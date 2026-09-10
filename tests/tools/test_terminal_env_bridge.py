"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os
import sys

import pytest

import tools.terminal_tool as terminal_tool
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an un-attempted bridge and clean mapped env."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    monkeypatch.setattr(
        terminal_tool, "_terminal_config_bridge_scope", None, raising=False
    )
    monkeypatch.setattr(
        terminal_tool, "_terminal_config_bridge_keys", set(), raising=False
    )
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


def test_env_is_preserved_when_config_has_no_terminal_section(monkeypatch):
    _write_config("agent:\n  max_turns: 100\n")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == "example.test"


# -- profile-multiplexing scope handoff (#94200) ------------------------------


def test_scope_change_rebridges_instead_of_inheriting_previous_profile(
    monkeypatch, tmp_path
):
    """Multiplexed gateway: a docker profile bridging first must not pin
    TERMINAL_ENV for a later local profile (#94200)."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (home_b / "config.yaml").write_text("terminal:\n  backend: local\n")

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token_a = set_hermes_home_override(str(home_a))
    try:
        config_a = terminal_tool._get_env_config()
        assert config_a["env_type"] == "docker"
        assert os.environ["TERMINAL_ENV"] == "docker"
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(home_b))
    try:
        config_b = terminal_tool._get_env_config()
    finally:
        reset_hermes_home_override(token_b)

    assert config_b["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"


def test_scope_change_without_terminal_section_drops_bridged_env(
    monkeypatch, tmp_path
):
    """The leak's worst shape: profile B has no terminal section, so the
    old `elif TERMINAL_ENV not in os.environ` branch kept profile A's
    docker selection forever. The bridge-introduced keys must be unset on
    handoff so B falls back to its own (default local) selection."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (home_b / "config.yaml").write_text("agent:\n  max_turns: 5\n")

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token_a = set_hermes_home_override(str(home_a))
    try:
        terminal_tool._get_env_config()
        assert os.environ["TERMINAL_ENV"] == "docker"
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(home_b))
    try:
        config_b = terminal_tool._get_env_config()
    finally:
        reset_hermes_home_override(token_b)

    assert config_b["env_type"] == "local"


def test_failed_rebridge_purges_previous_profile_before_latching(
    monkeypatch, tmp_path
):
    """If the incoming profile's bridge fails — e.g. the config import
    raises — the previous profile's bridged TERMINAL_* keys must already
    be gone: a broken config machinery must not re-leak the previous
    backend selection while the attempt flag latches the failure (#94200).
    The call degrades to the historical local default instead."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (home_b / "config.yaml").write_text("terminal:\n  backend: local\n")

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token_a = set_hermes_home_override(str(home_a))
    try:
        terminal_tool._get_env_config()
        assert os.environ["TERMINAL_ENV"] == "docker"
    finally:
        reset_hermes_home_override(token_a)

    # Break profile B's bridge where it hurts: the config import itself,
    # before the old code ever reached the key purge. Scoped so the
    # post-bridge imports in _get_env_config stay untouched.
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "hermes_cli.config", None)
        token_b = set_hermes_home_override(str(home_b))
        try:
            terminal_tool._ensure_terminal_env_bridged()
        finally:
            reset_hermes_home_override(token_b)

    # A's bridged key did not survive the failed re-bridge — the call
    # degrades to the ambient env (no TERMINAL_ENV = local default), not
    # to profile A's docker selection.
    assert "TERMINAL_ENV" not in os.environ
    # The failure still latches, keeping the one-shot semantics: a broken
    # config must not be re-attempted on every call within this scope.
    assert terminal_tool._terminal_config_bridge_attempted is True


def test_same_scope_keeps_one_shot_semantics(monkeypatch, tmp_path):
    """Within one profile the bridge still runs at most once (no per-call
    config re-reads)."""
    home = tmp_path / "single-profile"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: docker\n")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))

    import hermes_cli.config as cli_config

    calls = {"n": 0}
    real_apply = cli_config.apply_terminal_config_to_env

    def _counting_apply(env=None, override=False):
        calls["n"] += 1
        return real_apply(env=env, override=override)

    monkeypatch.setattr(cli_config, "apply_terminal_config_to_env", _counting_apply)
    try:
        terminal_tool._get_env_config()
        terminal_tool._get_env_config()
        terminal_tool._get_env_config()

        assert calls["n"] == 1
        assert os.environ["TERMINAL_ENV"] == "docker"
    finally:
        reset_hermes_home_override(token)


def test_handoff_never_unsets_shell_exported_terminal_vars(
    monkeypatch, tmp_path
):
    """Keys the user's shell/.env exported are not bridge-owned and must
    survive a scope handoff."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text("agent:\n  max_turns: 5\n")
    (home_b / "config.yaml").write_text("agent:\n  max_turns: 5\n")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "shell/image:9")

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token_a = set_hermes_home_override(str(home_a))
    try:
        terminal_tool._get_env_config()
    finally:
        reset_hermes_home_override(token_a)
    token_b = set_hermes_home_override(str(home_b))
    try:
        terminal_tool._get_env_config()
    finally:
        reset_hermes_home_override(token_b)

    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "shell/image:9"


# -- unified dashboard profile switcher (#98581) --------------------------------


def test_profile_switch_serves_each_profiles_terminal_backend(tmp_path):
    """A unified dashboard backend serves non-default profiles under a
    context-local HERMES_HOME override; each profile must execute commands
    with ITS configured terminal backend, not the launch profile's (#98581 —
    profiles with ``terminal.backend: docker`` ran on the host instead)."""
    launch_home = tmp_path / "root"
    docker_home = tmp_path / "root" / "profiles" / "web"
    docker_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("terminal:\n  backend: local\n")
    (docker_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  docker_image: sandbox/image:1\n"
    )

    launch_token = set_hermes_home_override(str(launch_home))
    try:
        assert terminal_tool._get_env_config()["env_type"] == "local"

        profile_token = set_hermes_home_override(str(docker_home))
        try:
            config = terminal_tool._get_env_config()
        finally:
            reset_hermes_home_override(profile_token)

        assert config["env_type"] == "docker"
        assert config["docker_image"] == "sandbox/image:1"
        assert os.environ["TERMINAL_ENV"] == "docker"

        # Back on the launch profile: the profile's docker selection must
        # not stick, and the launch backend applies again.
        assert terminal_tool._get_env_config()["env_type"] == "local"
        assert os.environ["TERMINAL_ENV"] == "local"
    finally:
        reset_hermes_home_override(launch_token)


def test_profile_switch_drops_launch_terminal_selection(tmp_path):
    """TERMINAL_* vars owned by the launch profile's terminal section must
    not survive a switch to a profile without one — the unconfigured profile
    falls back to the local default, not the launch profile's backend."""
    launch_home = tmp_path / "root"
    plain_home = tmp_path / "root" / "profiles" / "plain"
    plain_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (plain_home / "config.yaml").write_text("agent:\n  max_turns: 100\n")

    launch_token = set_hermes_home_override(str(launch_home))
    try:
        assert terminal_tool._get_env_config()["env_type"] == "docker"
        assert os.environ["TERMINAL_ENV"] == "docker"

        profile_token = set_hermes_home_override(str(plain_home))
        try:
            config = terminal_tool._get_env_config()
        finally:
            reset_hermes_home_override(profile_token)

        assert config["env_type"] == "local"
        assert os.environ["TERMINAL_ENV"] == "local"
    finally:
        reset_hermes_home_override(launch_token)


def test_handoff_drops_launcher_bridged_selection_before_first_call(
    monkeypatch, tmp_path
):
    """A launcher bridge (CLI ``env_mappings``) can write the launch home's
    selection BEFORE the first terminal-tool bridge call ever runs. Those
    vars belong to the launch home's terminal section all the same and must
    be dropped on handoff — ownership comes from the config section, not
    from what this bridge happened to introduce (#98581)."""
    launch_home = tmp_path / "root"
    plain_home = tmp_path / "root" / "profiles" / "plain"
    plain_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (plain_home / "config.yaml").write_text("agent:\n  max_turns: 100\n")
    # The CLI launcher bridged the launch home's selection into the env
    # before any terminal-tool call ran in this process.
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    launch_token = set_hermes_home_override(str(launch_home))
    try:
        assert terminal_tool._get_env_config()["env_type"] == "docker"

        profile_token = set_hermes_home_override(str(plain_home))
        try:
            config = terminal_tool._get_env_config()
        finally:
            reset_hermes_home_override(profile_token)

        assert config["env_type"] == "local"
        assert os.environ["TERMINAL_ENV"] == "local"
    finally:
        reset_hermes_home_override(launch_token)


def test_defaults_backfill_when_neither_config_nor_env_selects_backend():
    _write_config("{}\n")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"


def test_bridge_only_attempted_once(monkeypatch):
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

    assert len(calls) == 1


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
