"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, Thread

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
    with terminal_tool._env_lock:
        terminal_tool._active_environments.clear()
        terminal_tool._last_activity.clear()
        terminal_tool._environment_metadata.clear()
        terminal_tool._task_environment_keys.clear()
    with terminal_tool._creation_locks_lock:
        terminal_tool._creation_locks.clear()
        terminal_tool._creation_lock_users.clear()
        terminal_tool._creation_lock_retired.clear()


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


def test_lazy_environment_creation_uses_task_cwd_override(monkeypatch):
    """Lazy image bring-up must use the same per-task CWD as terminal calls."""
    _write_config(
        "terminal:\n"
        "  backend: ssh\n"
        "  cwd: '~'\n"
        "  ssh_host: example.test\n"
    )
    task_id = "lazy-cwd-task"
    terminal_tool.register_task_env_overrides(task_id, {"cwd": "/srv/repo"})
    captured = {}

    class FakeEnvironment:
        def cleanup(self):
            return None

    def fake_create_environment(**kwargs):
        captured.update(kwargs)
        return FakeEnvironment()

    monkeypatch.setattr(
        terminal_tool, "_create_environment", fake_create_environment,
    )

    try:
        assert terminal_tool.ensure_task_env(task_id) is not None
    finally:
        terminal_tool.clear_task_env_overrides(task_id)

    assert captured["cwd"] == "/srv/repo"
    assert captured["host_cwd"] is None


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


def test_raw_mpx_task_ids_are_still_namespaced_per_profile(tmp_path, monkeypatch):
    """A user task ID must never masquerade as an internal resolved key."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    secret_scope.set_multiplex_active(True)
    resolved = []
    for name in ("profile-a", "profile-b"):
        home = tmp_path / name
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            resolved.append(terminal_tool._resolve_container_task_id("mpx:attacker"))
        finally:
            reset_hermes_home_override(token)

    assert resolved[0] != resolved[1]
    assert all(key != "mpx:attacker" for key in resolved)


def test_isolated_workspace_task_ending_default_uses_its_own_mount(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: True)
    terminal_tool.register_task_env_overrides(
        "workspace:default",
        {"cwd": str(workspace), "cwd_source": "session"},
    )
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": "/global/workspace",
    }
    try:
        assert terminal_tool._resolve_task_host_cwd(
            config,
            "workspace:default",
        ) == str(workspace)
    finally:
        terminal_tool.clear_task_env_overrides("workspace:default")


def test_managed_terminal_backend_wins_in_snapshot_and_cache_identity(monkeypatch):
    from hermes_cli import managed_scope

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setattr(
        managed_scope,
        "load_managed_config",
        lambda: {"terminal": {"backend": "ssh"}},
    )
    raw = {"terminal": {"backend": "local"}}

    assert terminal_tool._terminal_env_snapshot(raw)["TERMINAL_ENV"] == "ssh"
    assert terminal_tool._terminal_backend_identity(raw)[1] == "ssh"


def test_stale_retirement_keeps_the_stable_creation_lock():
    task_id = "stable-lock-task"
    config_a = {"env_type": "local", "cwd": "/a"}
    config_b = {"env_type": "local", "cwd": "/b"}
    with terminal_tool._creation_locks_lock:
        lock = terminal_tool._creation_locks.setdefault(task_id, Lock())

    class Environment:
        def cleanup(self):
            return None

    terminal_tool._register_active_environment(
        task_id,
        Environment(),
        config_a,
        task_id,
    )
    try:
        terminal_tool._retire_stale_environment_for_config(
            task_id,
            task_id,
            config_b,
        )
        with terminal_tool._creation_locks_lock:
            assert terminal_tool._creation_locks.setdefault(
                task_id,
                Lock(),
            ) is lock
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
            terminal_tool._last_activity.pop(task_id, None)
            terminal_tool._environment_metadata.pop(task_id, None)
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.pop(task_id, None)


def test_stale_cleanup_blocks_replacement_creation(monkeypatch):
    import tools.code_execution_tool as code_execution_tool

    task_id = "stale-cleanup-race"
    cleanup_started = Event()
    release_cleanup = Event()
    creation_started = Event()
    config_a = {"env_type": "local", "cwd": "/a", "timeout": 30}
    config_b = {"env_type": "local", "cwd": "/b", "timeout": 30}
    created = []
    results = []

    class OldEnvironment:
        def cleanup(self):
            cleanup_started.set()
            release_cleanup.wait(timeout=2)

    class NewEnvironment:
        pass

    def create_environment(**_kwargs):
        creation_started.set()
        env = NewEnvironment()
        created.append(env)
        return env

    terminal_tool._register_active_environment(
        task_id, OldEnvironment(), config_a, task_id,
    )
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config_b)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    def get_environment():
        results.append(code_execution_tool._get_or_create_env(task_id)[0])

    first = Thread(target=get_environment)
    second = Thread(target=get_environment)
    try:
        first.start()
        assert cleanup_started.wait(timeout=1)
        second.start()
        assert not creation_started.wait(timeout=0.2)
        release_cleanup.set()
        first.join(timeout=2)
        second.join(timeout=2)
        assert not first.is_alive()
        assert not second.is_alive()
        assert len(created) == 1
        assert results == [created[0], created[0]]
    finally:
        release_cleanup.set()
        first.join(timeout=2)
        second.join(timeout=2)
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
            terminal_tool._last_activity.pop(task_id, None)
            terminal_tool._forget_environment_key(task_id)
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.pop(task_id, None)


def test_degraded_eviction_keeps_a_held_creation_lock():
    task_id = "held-eviction-lock"
    lock = Lock()
    lock.acquire()
    eviction_done = Event()
    with terminal_tool._creation_locks_lock:
        terminal_tool._creation_locks[task_id] = lock
    worker = Thread(
        target=lambda: (
            terminal_tool._evict_environment_for_task(task_id),
            eviction_done.set(),
        )
    )
    try:
        worker.start()
        assert not eviction_done.wait(timeout=0.2)
        with terminal_tool._creation_locks_lock:
            assert terminal_tool._creation_locks.get(task_id) is lock
        lock.release()
        worker.join(timeout=2)
        assert eviction_done.is_set()
        with terminal_tool._creation_locks_lock:
            assert task_id not in terminal_tool._creation_locks
    finally:
        if lock.locked():
            lock.release()
        worker.join(timeout=2)
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.pop(task_id, None)


def test_execute_code_remote_creation_uses_full_container_policy(monkeypatch):
    import tools.code_execution_tool as code_execution_tool

    task_id = "managed-modal-policy"
    captured = {}
    config = {
        "env_type": "modal",
        "modal_image": "python:3.11",
        "modal_mode": "managed",
        "cwd": "/workspace",
        "timeout": 30,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 8192,
        "container_persistent": True,
        "docker_forward_env": ["HTTP_PROXY"],
        "docker_env": {"MODE": "test"},
        "docker_extra_args": ["--cap-drop=ALL"],
        "docker_mount_cwd_to_workspace": True,
        "docker_persist_across_processes": False,
        "docker_network": False,
    }

    class Environment:
        pass

    def create_environment(**kwargs):
        captured.update(kwargs)
        return Environment()

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    try:
        code_execution_tool._get_or_create_env(task_id)
        expected = terminal_tool._container_config_from_config(config)
        assert captured["container_config"] == expected
        assert captured["container_config"]["modal_mode"] == "managed"
        assert captured["container_config"]["docker_network"] is False
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
            terminal_tool._last_activity.pop(task_id, None)
            terminal_tool._forget_environment_key(task_id)
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks.pop(task_id, None)


def test_cleanup_vm_clears_canonical_task_overrides():
    task_id = "cleanup-leak"
    terminal_tool.register_task_env_overrides(
        task_id,
        {"env_type": "docker", "docker_image": "private/image"},
    )
    terminal_tool.cleanup_vm(task_id)
    assert terminal_tool.resolve_task_overrides(task_id) == {}


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


def test_child_first_environment_creation_uses_parent_alias_overrides(monkeypatch):
    import tools.code_execution_tool as code_execution_tool

    config = terminal_tool._get_env_config(
        {
            "terminal": {
                "backend": "docker",
                "container_persistent": False,
            }
        }
    )
    config["container_persistent"] = False
    created = {}

    class FakeEnvironment:
        def cleanup(self):
            pass

    def fake_create_environment(**kwargs):
        created.update(kwargs)
        return FakeEnvironment()

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda *_args: config)
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    secret_scope.set_multiplex_active(True)
    terminal_tool.register_task_env_overrides(
        "parent-task",
        {
            "docker_image": "parent/image:1",
            "cwd": "/workspace/parent",
        },
    )
    terminal_tool.register_container_alias("child-task", "parent-task")
    try:
        code_execution_tool._get_or_create_env("child-task")
    finally:
        terminal_tool.cleanup_vm("parent-task")
        terminal_tool.clear_task_env_overrides("parent-task")
        terminal_tool.clear_task_env_overrides("child-task")
        secret_scope.set_multiplex_active(False)

    assert created["task_id"].endswith(":parent-task")
    assert created["image"] == "parent/image:1"
    assert created["cwd"] == "/workspace/parent"


def test_child_cleanup_does_not_destroy_parent_aliased_environment(monkeypatch):
    config = terminal_tool._get_env_config(
        {
            "terminal": {
                "backend": "docker",
                "container_persistent": False,
            }
        }
    )
    config["container_persistent"] = False
    cleaned = []

    class FakeEnvironment:
        def cleanup(self):
            cleaned.append(True)

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda *_args: config)
    secret_scope.set_multiplex_active(True)
    terminal_tool.register_container_alias("child-task", "parent-task")
    environment_key = terminal_tool._resolve_container_task_id("child-task")
    terminal_tool._register_active_environment(
        environment_key,
        FakeEnvironment(),
        config,
        "child-task",
    )
    try:
        terminal_tool.cleanup_vm("child-task")
        with terminal_tool._env_lock:
            assert environment_key in terminal_tool._active_environments
        assert cleaned == []

        terminal_tool.cleanup_vm("parent-task")
        with terminal_tool._env_lock:
            assert environment_key not in terminal_tool._active_environments
        assert cleaned == [True]
    finally:
        secret_scope.set_multiplex_active(False)
        with terminal_tool._env_lock:
            terminal_tool._active_environments.clear()
            terminal_tool._last_activity.clear()
            terminal_tool._environment_metadata.clear()
            terminal_tool._task_environment_keys.clear()
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


def test_nonmultiplex_config_change_retires_stale_environment(monkeypatch):
    import tools.code_execution_tool as code_execution_tool

    cleaned = []

    class FakeEnvironment:
        def __init__(self, backend):
            self.backend = backend

        def cleanup(self):
            cleaned.append(self.backend)

    def fake_create_environment(*, env_type, **_kwargs):
        return FakeEnvironment(env_type)

    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    _write_config("terminal:\n  backend: local\n")

    local_env, local_backend = code_execution_tool._get_or_create_env("reload-task")
    _write_config(
        "terminal:\n"
        "  backend: ssh\n"
        "  ssh_host: changed.example\n"
        "  ssh_user: changed-user\n"
    )
    ssh_env, ssh_backend = code_execution_tool._get_or_create_env("reload-task")
    _write_config("terminal:\n  backend: local\n")
    second_local_env, second_local_backend = (
        code_execution_tool._get_or_create_env("reload-task")
    )

    assert local_backend == "local"
    assert ssh_backend == "ssh"
    assert second_local_backend == "local"
    assert ssh_env is not local_env
    assert second_local_env is not ssh_env
    assert cleaned == ["local", "ssh"]


def test_cleanup_vm_retires_all_profile_session_namespace_versions(
    tmp_path,
    monkeypatch,
):
    from tools import file_tools

    home = tmp_path / "profile"
    home.mkdir()
    cleaned = []

    class FakeEnvironment:
        def __init__(self, name):
            self.name = name

        def cleanup(self):
            cleaned.append(self.name)

    monkeypatch.setattr(terminal_tool, "_profile_home_key", lambda: str(home))
    for key, backend in (
        ("mpx:old:default", "local"),
        ("mpx:new:default", "ssh"),
    ):
        config = terminal_tool._get_env_config(
            {
                "terminal": {
                    "backend": backend,
                    "ssh_host": "changed.example",
                    "ssh_user": "changed-user",
                }
            }
        )
        terminal_tool._register_active_environment(
            key,
            FakeEnvironment(key),
            config,
            "shared-session",
        )
        with terminal_tool._creation_locks_lock:
            terminal_tool._creation_locks[key] = Lock()
        with file_tools._file_ops_lock:
            file_tools._file_ops_cache[key] = object()

    terminal_tool.cleanup_vm("shared-session", profile_home=str(home))

    assert sorted(cleaned) == ["mpx:new:default", "mpx:old:default"]
    with terminal_tool._env_lock:
        assert not terminal_tool._active_environments
        assert not terminal_tool._task_environment_keys
    with terminal_tool._creation_locks_lock:
        assert "mpx:old:default" not in terminal_tool._creation_locks
        assert "mpx:new:default" not in terminal_tool._creation_locks
    with file_tools._file_ops_lock:
        assert "mpx:old:default" not in file_tools._file_ops_cache
        assert "mpx:new:default" not in file_tools._file_ops_cache


def test_cleanup_vm_resolves_child_alias_in_explicit_owner_profile(tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in homes:
        home.mkdir()

    secret_scope.set_multiplex_active(True)
    try:
        for home, parent in zip(homes, ("parent-a", "parent-b")):
            token = set_hermes_home_override(home)
            try:
                terminal_tool.register_container_alias("shared-child", parent)
            finally:
                reset_hermes_home_override(token)

        token = set_hermes_home_override(homes[1])
        try:
            terminal_tool.cleanup_vm(
                "shared-child",
                profile_home=str(homes[0]),
            )
            assert terminal_tool._resolve_container_alias("shared-child") == "parent-b"
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(homes[0])
        try:
            assert terminal_tool._resolve_container_alias("shared-child") == "shared-child"
        finally:
            reset_hermes_home_override(token)
    finally:
        with terminal_tool._container_alias_lock:
            terminal_tool._container_aliases.clear()


def test_explicit_profile_cleanup_waits_for_inflight_environment_publish(
    tmp_path,
):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    home = tmp_path / "profile"
    home.mkdir()
    cleaned = Event()
    creator_has_lock = Event()
    permit_register = Event()

    class FakeEnvironment:
        def cleanup(self):
            cleaned.set()

    secret_scope.set_multiplex_active(True)
    token = set_hermes_home_override(home)
    try:
        config = terminal_tool._get_env_config(
            {"terminal": {"backend": "local"}}
        )
        env_key = terminal_tool._resolve_container_task_id("race-session")

        def creator():
            with terminal_tool._task_creation_lock(env_key):
                creator_has_lock.set()
                permit_register.wait(timeout=2)
                terminal_tool._register_active_environment(
                    env_key,
                    FakeEnvironment(),
                    config,
                    "race-session",
                )

        def release_after_cleanup_waits():
            for _ in range(100):
                with terminal_tool._creation_locks_lock:
                    if terminal_tool._creation_lock_users.get(env_key) == 2:
                        permit_register.set()
                        return
                permit_register.wait(0.005)
            permit_register.set()

        creator_thread = Thread(target=creator)
        creator_thread.start()
        assert creator_has_lock.wait(timeout=2)
        release_thread = Thread(target=release_after_cleanup_waits)
        release_thread.start()

        terminal_tool.cleanup_vm(
            "race-session",
            profile_home=str(home),
        )

        creator_thread.join(timeout=2)
        release_thread.join(timeout=2)
        assert not creator_thread.is_alive()
        assert not release_thread.is_alive()
        assert cleaned.is_set()
        with terminal_tool._env_lock:
            assert env_key not in terminal_tool._active_environments
    finally:
        reset_hermes_home_override(token)


def test_cleanup_inactive_envs_uses_each_environment_lifetime(monkeypatch):
    cleaned = []

    class FakeEnvironment:
        def __init__(self, name):
            self.name = name

        def cleanup(self):
            cleaned.append(self.name)

    monkeypatch.setattr(terminal_tool.time, "time", lambda: 50.0)
    base_config = terminal_tool._get_env_config({"terminal": {"backend": "local"}})
    short_config = {**base_config, "lifetime_seconds": 10}
    long_config = {**base_config, "lifetime_seconds": 100}
    terminal_tool._register_active_environment(
        "short-env", FakeEnvironment("short"), short_config, "short-task",
    )
    terminal_tool._register_active_environment(
        "long-env", FakeEnvironment("long"), long_config, "long-task",
    )
    with terminal_tool._env_lock:
        terminal_tool._last_activity["short-env"] = 0.0
        terminal_tool._last_activity["long-env"] = 0.0

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("cleanup must not read an unscoped profile config")
        ),
    )
    worker = Thread(target=terminal_tool._cleanup_inactive_envs)
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert cleaned == ["short"]
    with terminal_tool._env_lock:
        assert "short-env" not in terminal_tool._active_environments
        assert "long-env" in terminal_tool._active_environments
