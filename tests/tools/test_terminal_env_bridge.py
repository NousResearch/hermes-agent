"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

import tools.terminal_tool as terminal_tool
from agent import secret_scope
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an un-attempted bridge and clean mapped env."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_CWD",
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_SSH_HOST",
        "TERMINAL_SSH_USER",
    ):
        monkeypatch.delenv(name, raising=False)
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


def _write_config(text: str) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


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


def test_profile_scoped_terminal_snapshots_do_not_share_process_env(
    tmp_path,
    monkeypatch,
):
    """Concurrent profile turns must resolve their own configured backend."""
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = {}
    profile_configs = {
        "local-profile": (
            "terminal:\n"
            "  backend: local\n"
            "  cwd: /tmp/local-profile\n"
        ),
        "ssh-profile": (
            "terminal:\n"
            "  backend: ssh\n"
            "  cwd: '~'\n"
            "  ssh_host: ssh-profile.example\n"
            "  ssh_user: profile-user\n"
        ),
    }
    for name, text in profile_configs.items():
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            text,
            encoding="utf-8",
        )
        homes[name] = home

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "stale-process.example")
    monkeypatch.setenv("TERMINAL_SSH_USER", "stale-user")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    secret_scope.set_multiplex_active(True)
    barrier = Barrier(2)

    def resolve(profile_name):
        token = set_hermes_home_override(homes[profile_name])
        try:
            barrier.wait(timeout=2)
            config = terminal_tool._get_env_config()
            return (
                config["env_type"],
                config["cwd"],
                config["ssh_host"],
                config["ssh_user"],
            )
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(resolve, "local-profile")
        ssh_future = pool.submit(resolve, "ssh-profile")

    assert local_future.result()[:2] == ("local", "/tmp/local-profile")
    assert ssh_future.result() == (
        "ssh",
        "~",
        "ssh-profile.example",
        "profile-user",
    )
    assert os.environ["TERMINAL_ENV"] == "local"
    assert os.environ["TERMINAL_SSH_HOST"] == "stale-process.example"


def test_multiplex_profiles_do_not_reuse_environment_cache_entries(
    tmp_path,
    monkeypatch,
):
    """A routed profile must not inherit another profile's cached backend."""
    import tools.code_execution_tool as code_execution_tool
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = {}
    for name, text in {
        "local-profile": "terminal:\n  backend: local\n",
        "ssh-profile": (
            "terminal:\n"
            "  backend: ssh\n"
            "  ssh_host: ssh-profile.example\n"
            "  ssh_user: profile-user\n"
        ),
    }.items():
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(text, encoding="utf-8")
        homes[name] = home

    created = []

    class FakeEnvironment:
        def __init__(self, backend):
            self.backend = backend

    def fake_create_environment(*, env_type, **_kwargs):
        env = FakeEnvironment(env_type)
        created.append(env)
        return env

    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    secret_scope.set_multiplex_active(True)

    def resolve(profile_name):
        token = set_hermes_home_override(homes[profile_name])
        try:
            return code_execution_tool._get_or_create_env("shared-task")
        finally:
            reset_hermes_home_override(token)

    try:
        local_env, local_backend = resolve("local-profile")
        ssh_env, ssh_backend = resolve("ssh-profile")
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.clear()
            terminal_tool._last_activity.clear()
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.clear()

    assert local_backend == "local"
    assert ssh_backend == "ssh"
    assert local_env is not ssh_env
    assert [env.backend for env in created] == ["local", "ssh"]


def test_cleanup_vm_resolves_the_active_profile_namespace(tmp_path, monkeypatch):
    """Session teardown must remove the profile-scoped cached environment."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "ssh-profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: ssh\n"
        "  ssh_host: ssh-profile.example\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    secret_scope.set_multiplex_active(True)

    cleaned = []

    class FakeEnvironment:
        def cleanup(self):
            cleaned.append(True)

    token = set_hermes_home_override(home)
    effective_task_id = terminal_tool._resolve_container_task_id("shared-task")
    try:
        with terminal_tool._env_lock:
            terminal_tool._active_environments[effective_task_id] = FakeEnvironment()
            terminal_tool._last_activity[effective_task_id] = 1.0
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks[effective_task_id] = Lock()

        terminal_tool.cleanup_vm("shared-task")

        with terminal_tool._env_lock:
            assert effective_task_id not in terminal_tool._active_environments
            assert effective_task_id not in terminal_tool._last_activity
        with terminal_tool._creation_locks_lock:
            assert effective_task_id not in terminal_tool._creation_locks
        assert cleaned == [True]
    finally:
        reset_hermes_home_override(token)
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(effective_task_id, None)
            terminal_tool._last_activity.pop(effective_task_id, None)
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.pop(effective_task_id, None)


def test_multiplex_profiles_do_not_share_session_terminal_state(tmp_path, monkeypatch):
    """Cwd, task overrides, and child aliases belong to one routed profile."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = []
    for name in ("profile-a", "profile-b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            "terminal:\n  backend: local\n",
            encoding="utf-8",
        )
        homes.append(home)

    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    secret_scope.set_multiplex_active(True)

    try:
        token = set_hermes_home_override(homes[0])
        try:
            terminal_tool.register_task_env_overrides(
                "shared-task",
                {"cwd": "/profile-a", "docker_image": "profile-a:image"},
            )
            terminal_tool.record_session_cwd("shared-task", "/profile-a/live")
            terminal_tool.register_container_alias("shared-child", "profile-a-parent")
            effective_task_id = terminal_tool._resolve_container_task_id("shared-task")
            assert terminal_tool._has_isolation_overrides(effective_task_id)
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(homes[1])
        try:
            assert terminal_tool.resolve_task_overrides("shared-task") == {}
            assert terminal_tool.get_session_cwd("shared-task") is None
            assert terminal_tool._resolve_container_alias("shared-child") == "shared-child"
        finally:
            reset_hermes_home_override(token)
    finally:
        terminal_tool._task_env_overrides.clear()
        with terminal_tool._session_cwd_lock:
            terminal_tool._session_cwd.clear()
        with terminal_tool._container_alias_lock:
            terminal_tool._container_aliases.clear()


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
