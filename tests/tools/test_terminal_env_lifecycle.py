"""Tests for terminal environment config, creation, lifecycle, and cleanup.

Covers ``tools/terminal_tool.py``:
- ``_get_env_config``                (config resolution from TERMINAL_* env vars)
- ``_ssh_config_from_config``        (SSH connection dict extraction)
- ``_container_config_from_config``  (container resource dict extraction)
- ``_create_environment``            (backend selection + unknown-type ValueError)
- ``ensure_task_env``                (lazy create / cache / failure)
- ``is_persistent_env``              (persistence predicate)
- ``cleanup_all_environments``       (global teardown)
- ``cleanup_vm``                     (per-task teardown + force_remove dispatch)
- ``_cleanup_inactive_envs``         (idle reaper)
- ``_atexit_cleanup``                (process-exit teardown)
- ``_evict_environment_for_task``    (drop a degraded env)
- ``register_task_env_overrides`` / ``clear_task_env_overrides``
- ``register_container_alias`` / ``_resolve_container_alias``

The module reads ALL of its settings from ``os.environ`` and keeps a set of
process-global registries (``_active_environments``, ``_last_activity``,
``_creation_locks``, ``_task_env_overrides``, ``_container_aliases``,
``_session_cwd``).  The ``_hermetic_env_state`` autouse fixture resets those
registries around every test so the shared interpreter never leaks state
between cases, and the config-bridge is stubbed so a developer's
``config.yaml`` cannot perturb TERMINAL_* results.
"""

import os
import sys
import threading
import time
import types

import pytest

import tools.terminal_tool as terminal_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DEFAULT_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"


def _stub_config_bridge(monkeypatch):
    """Prevent the one-shot config.yaml -> env bridge from mutating os.environ."""
    monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", lambda: None)


class _CleanupRecorder:
    """Double for an environment that records how it was torn down."""

    def __init__(self, accept_force_remove=False):
        self.cleaned = False
        self.stopped = False
        self.terminated = False
        self.force_remove = None
        self._accept_force_remove = accept_force_remove

    def cleanup(self, **kwargs):
        self.cleaned = True
        if self._accept_force_remove:
            self.force_remove = kwargs.get("force_remove")

    def stop(self):
        self.stopped = True

    def terminate(self):
        self.terminated = True


@pytest.fixture(autouse=True)
def _hermetic_env_state(monkeypatch):
    """Reset the module's process-global registries around each test.

    Each test file gets a fresh interpreter, but tests *within* this file
    share one.  These dicts are mutated by the lifecycle functions under test,
    so any residue would silently change later assertions (e.g. a leftover
    ``_task_env_overrides`` entry flipping ``_resolve_container_task_id``).
    """
    registries = [
        "_active_environments",
        "_last_activity",
        "_creation_locks",
        "_task_env_overrides",
        "_container_aliases",
        "_session_cwd",
    ]
    # Snapshot the original objects so monkeypatch.setattr in a test restores
    # them afterwards, and clear their contents both sides of each test.
    originals = {name: getattr(terminal_tool, name) for name in registries}
    monkeypatch.setattr(terminal_tool, "_container_alias_lock", threading.Lock())
    for name in registries:
        originals[name].clear()
    # Neutralise any host TERMINAL_* env so config resolution is deterministic.
    for key in list(os.environ):
        if key.startswith("TERMINAL_"):
            monkeypatch.delenv(key, raising=False)
    yield
    for name in registries:
        getattr(terminal_tool, name).clear()
    # Restore any genuine objects a test swapped in via setattr anyway.
    monkeypatch.setattr(terminal_tool, "_container_alias_lock", threading.Lock())


def _fake_process_registry(monkeypatch):
    """Install a no-op process_registry so `_cleanup_inactive_envs` uses it."""
    fake = types.ModuleType("tools.process_registry")

    class _Registry:
        @staticmethod
        def has_active_processes(task_id):
            return False

    fake.process_registry = _Registry()
    monkeypatch.setitem(sys.modules, "tools.process_registry", fake)


# ---------------------------------------------------------------------------
# _get_env_config
# ---------------------------------------------------------------------------
class TestGetEnvConfig:
    def test_defaults_local(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        for key in list(os.environ):
            if key.startswith("TERMINAL_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(terminal_tool, "_safe_getcwd", lambda: "/host/workdir")

        cfg = terminal_tool._get_env_config()

        assert cfg["env_type"] == "local"
        assert cfg["docker_image"] == _DEFAULT_IMAGE
        assert cfg["singularity_image"] == f"docker://{_DEFAULT_IMAGE}"
        assert cfg["modal_image"] == _DEFAULT_IMAGE
        assert cfg["daytona_image"] == _DEFAULT_IMAGE
        assert cfg["vercel_runtime"] == ""
        assert cfg["cwd"] == "/host/workdir"
        assert cfg["host_cwd"] is None
        assert cfg["timeout"] == 180
        assert cfg["lifetime_seconds"] == 300
        assert cfg["container_cpu"] == 1.0
        assert cfg["container_memory"] == 5120
        assert cfg["container_disk"] == 51200
        assert cfg["container_persistent"] is True
        assert cfg["local_persistent"] is False
        assert cfg["ssh_persistent"] is True
        assert cfg["ssh_host"] == ""
        assert cfg["ssh_user"] == ""
        assert cfg["ssh_port"] == 22
        assert cfg["ssh_key"] == ""
        assert cfg["docker_volumes"] == []
        assert cfg["docker_env"] == {}
        assert cfg["docker_extra_args"] == []
        assert cfg["docker_forward_env"] == []
        assert cfg["docker_shm_size"] == "1g"
        assert cfg["docker_network"] is True
        assert cfg["docker_run_as_host_user"] is False
        assert cfg["docker_mount_cwd_to_workspace"] is False
        assert cfg["docker_persist_across_processes"] is True
        assert cfg["docker_orphan_reaper"] is True

    def test_env_type_from_env(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        cfg = terminal_tool._get_env_config()
        assert cfg["env_type"] == "docker"
        # Container backends default to /root when no TERMINAL_CWD set.
        assert cfg["cwd"] == "/root"

    def test_docker_image_override(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "myrepo/edge:latest")
        cfg = terminal_tool._get_env_config()
        assert cfg["docker_image"] == "myrepo/edge:latest"

    def test_cwd_env_override(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", "/custom/workspace")
        cfg = terminal_tool._get_env_config()
        assert cfg["cwd"] == "/custom/workspace"

    def test_container_backend_rejects_host_cwd(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CWD", "/home/vagrant/proj")
        cfg = terminal_tool._get_env_config()
        # A host path cannot be a container workdir -> fall back to /root.
        assert cfg["cwd"] == "/root"

    def test_ssh_env_fields(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "sandbox.example.com")
        monkeypatch.setenv("TERMINAL_SSH_USER", "ubuntu")
        monkeypatch.setenv("TERMINAL_SSH_PORT", "2222")
        monkeypatch.setenv("TERMINAL_SSH_KEY", "/home/user/.ssh/id_ed25519")
        cfg = terminal_tool._get_env_config()
        assert cfg["ssh_host"] == "sandbox.example.com"
        assert cfg["ssh_user"] == "ubuntu"
        assert cfg["ssh_port"] == 2222
        assert cfg["ssh_key"] == "/home/user/.ssh/id_ed25519"
        # SSH default cwd is the remote home.
        assert cfg["cwd"] == "~"

    def test_docker_volumes_json_parsed(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            '[{"host_path": "/data", "container_path": "/workspace", "read_only": false}]',
        )
        cfg = terminal_tool._get_env_config()
        assert cfg["docker_volumes"] == [
            {"host_path": "/data", "container_path": "/workspace", "read_only": False}
        ]

    def test_docker_env_json_parsed(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_ENV", '{"A": "1", "B": "2"}')
        cfg = terminal_tool._get_env_config()
        assert cfg["docker_env"] == {"A": "1", "B": "2"}

    def test_timeout_env(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_TIMEOUT", "90")
        cfg = terminal_tool._get_env_config()
        assert cfg["timeout"] == 90

    def test_invalid_timeout_raises(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_TIMEOUT", "5m")
        with pytest.raises(ValueError):
            terminal_tool._get_env_config()

    def test_invalid_docker_volumes_raises(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "not-json")
        with pytest.raises(ValueError):
            terminal_tool._get_env_config()

    def test_container_persistent_bool(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
        cfg = terminal_tool._get_env_config()
        assert cfg["container_persistent"] is False

    def test_ssh_persistent_env_override(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "false")
        cfg = terminal_tool._get_env_config()
        assert cfg["ssh_persistent"] is False

    def test_local_persistent_env(self, monkeypatch):
        _stub_config_bridge(monkeypatch)
        monkeypatch.setenv("TERMINAL_LOCAL_PERSISTENT", "true")
        cfg = terminal_tool._get_env_config()
        assert cfg["local_persistent"] is True


# ---------------------------------------------------------------------------
# _ssh_config_from_config
# ---------------------------------------------------------------------------
class TestSshConfigFromConfig:
    def test_extracts_fields(self):
        result = terminal_tool._ssh_config_from_config(
            {
                "ssh_host": "h",
                "ssh_user": "u",
                "ssh_port": 2222,
                "ssh_key": "/k",
                "ssh_persistent": True,
            }
        )
        assert result == {
            "host": "h",
            "user": "u",
            "port": 2222,
            "key": "/k",
            "persistent": True,
        }

    def test_defaults(self):
        result = terminal_tool._ssh_config_from_config({})
        assert result == {
            "host": "",
            "user": "",
            "port": 22,
            "key": "",
            "persistent": False,
        }


# ---------------------------------------------------------------------------
# _container_config_from_config
# ---------------------------------------------------------------------------
class TestContainerConfigFromConfig:
    def test_extracts_fields(self):
        config = {
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 100,
            "container_persistent": False,
            "modal_mode": "direct",
            "vercel_runtime": "node18",
            "docker_volumes": [{"host_path": "/data"}],
            "docker_mount_cwd_to_workspace": True,
            "docker_forward_env": [{"name": "A"}],
            "docker_env": {"A": "1"},
            "docker_run_as_host_user": True,
            "docker_extra_args": ["--network", "host"],
            "docker_shm_size": "2g",
            "docker_network": False,
            "docker_persist_across_processes": False,
            "docker_shared_container_key": "shared:key",
            "docker_orphan_reaper": False,
        }
        result = terminal_tool._container_config_from_config(config)
        assert result["container_cpu"] == 2
        assert result["container_memory"] == 4096
        assert result["container_disk"] == 100
        assert result["container_persistent"] is False
        assert result["modal_mode"] == "direct"
        assert result["vercel_runtime"] == "node18"
        assert result["docker_volumes"] == [{"host_path": "/data"}]
        assert result["docker_mount_cwd_to_workspace"] is True
        assert result["docker_forward_env"] == [{"name": "A"}]
        assert result["docker_env"] == {"A": "1"}
        assert result["docker_run_as_host_user"] is True
        assert result["docker_extra_args"] == ["--network", "host"]
        assert result["docker_shm_size"] == "2g"
        assert result["docker_network"] is False
        assert result["docker_persist_across_processes"] is False
        assert result["docker_shared_container_key"] == "shared:key"
        assert result["docker_orphan_reaper"] is False

    def test_defaults(self):
        result = terminal_tool._container_config_from_config({})
        assert result["container_cpu"] == 1
        assert result["container_memory"] == 5120
        assert result["container_disk"] == 51200
        assert result["container_persistent"] is True
        assert result["modal_mode"] == "auto"
        assert result["vercel_runtime"] == ""
        assert result["docker_volumes"] == []
        assert result["docker_mount_cwd_to_workspace"] is False
        assert result["docker_forward_env"] == []
        assert result["docker_env"] == {}
        assert result["docker_run_as_host_user"] is False
        assert result["docker_extra_args"] == []
        assert result["docker_shm_size"] == "1g"
        assert result["docker_network"] is True
        assert result["docker_persist_across_processes"] is True
        assert result["docker_shared_container_key"] == ""
        assert result["docker_orphan_reaper"] is True


# ---------------------------------------------------------------------------
# _create_environment
# ---------------------------------------------------------------------------
class TestCreateEnvironment:
    def test_create_local(self, monkeypatch):
        calls = {}

        class _FakeLocal:
            def __init__(self, cwd, timeout):
                calls["cwd"] = cwd
                calls["timeout"] = timeout

        monkeypatch.setattr(terminal_tool, "_LocalEnvironment", _FakeLocal)
        env = terminal_tool._create_environment(
            "local", image="", cwd="/workspace", timeout=42
        )
        assert isinstance(env, _FakeLocal)
        assert calls == {"cwd": "/workspace", "timeout": 42}

    def test_create_docker(self, monkeypatch):
        calls = {}

        class _FakeDocker:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(terminal_tool, "_DockerEnvironment", _FakeDocker)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
        monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: False)
        container_config = {
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 100,
            "container_persistent": False,
            "docker_volumes": [{"host_path": "/data"}],
            "docker_mount_cwd_to_workspace": True,
            "docker_forward_env": [{"name": "A"}],
            "docker_env": {"A": "1"},
            "docker_run_as_host_user": True,
            "docker_network": False,
            "docker_extra_args": ["--network", "host"],
            "docker_persist_across_processes": False,
            "docker_shared_container_key": "shared:key",
            "docker_shm_size": "2g",
        }
        env = terminal_tool._create_environment(
            "docker",
            image="img:tag",
            cwd="/workspace",
            timeout=30,
            container_config=container_config,
            task_id="task1",
            host_cwd="/host",
        )
        assert isinstance(env, _FakeDocker)
        assert calls["image"] == "img:tag"
        assert calls["cwd"] == "/workspace"
        assert calls["timeout"] == 30
        assert calls["cpu"] == 2
        assert calls["memory"] == 4096
        assert calls["disk"] == 100
        assert calls["persistent_filesystem"] is False
        assert calls["task_id"] == "task1"
        assert calls["host_cwd"] == "/host"
        assert calls["auto_mount_cwd"] is True
        assert calls["forward_env"] == [{"name": "A"}]
        assert calls["env"] == {"A": "1"}
        assert calls["run_as_host_user"] is True
        assert calls["network"] is False
        assert calls["extra_args"] == ["--network", "host"]
        assert calls["persist_across_processes"] is False
        assert calls["shared_container_key"] == "shared:key"
        assert calls["shm_size"] == "2g"

    def test_create_docker_defaults(self, monkeypatch):
        calls = {}

        class _FakeDocker:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(terminal_tool, "_DockerEnvironment", _FakeDocker)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
        monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: False)
        terminal_tool._create_environment(
            "docker", image="img", cwd="/workspace", timeout=30, task_id="default"
        )
        assert calls["cpu"] == 1
        assert calls["memory"] == 5120
        assert calls["disk"] == 51200
        assert calls["persistent_filesystem"] is True
        assert calls["network"] is True
        assert calls["run_as_host_user"] is False
        assert calls["persist_across_processes"] is True
        assert calls["shm_size"] == "1g"
        assert calls["auto_mount_cwd"] is False

    def test_create_ssh(self, monkeypatch):
        calls = {}

        class _FakeSSH:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(terminal_tool, "_SSHEnvironment", _FakeSSH)
        ssh_config = {
            "host": "h",
            "user": "u",
            "port": 2222,
            "key": "/k",
            "persistent": True,
        }
        env = terminal_tool._create_environment(
            "ssh", image="", cwd="/home/u", timeout=30, ssh_config=ssh_config
        )
        assert isinstance(env, _FakeSSH)
        assert calls["host"] == "h"
        assert calls["user"] == "u"
        assert calls["port"] == 2222
        assert calls["key_path"] == "/k"
        assert calls["cwd"] == "/home/u"
        assert calls["timeout"] == 30

    def test_create_ssh_missing_config_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            terminal_tool._create_environment(
                "ssh", image="", cwd="/home/u", timeout=30, ssh_config={}
            )

    def test_create_ssh_missing_host_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            terminal_tool._create_environment(
                "ssh",
                image="",
                cwd="/home/u",
                timeout=30,
                ssh_config={"user": "u"},
            )

    def test_create_singularity(self, monkeypatch):
        calls = {}

        class _FakeSingularity:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(terminal_tool, "_SingularityEnvironment", _FakeSingularity)
        terminal_tool._create_environment(
            "singularity",
            image="img",
            cwd="/w",
            timeout=30,
            container_config={"container_cpu": 2, "container_memory": 4096, "container_disk": 100, "container_persistent": False},
            task_id="task1",
        )
        assert calls["image"] == "img"
        assert calls["cpu"] == 2
        assert calls["memory"] == 4096
        assert calls["disk"] == 100
        assert calls["persistent_filesystem"] is False
        assert calls["task_id"] == "task1"

    def test_create_modal(self, monkeypatch):
        calls = {}

        class _FakeModal:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        monkeypatch.setattr(terminal_tool, "_ModalEnvironment", _FakeModal)
        monkeypatch.setattr(
            terminal_tool,
            "_get_modal_backend_state",
            lambda mode: {"selected_backend": "direct", "mode": "direct"},
        )
        terminal_tool._create_environment(
            "modal",
            image="img",
            cwd="/w",
            timeout=30,
            container_config={"container_cpu": 2, "container_memory": 4096, "container_persistent": True},
            task_id="task1",
        )
        assert calls["image"] == "img"
        assert calls["modal_sandbox_kwargs"]["cpu"] == 2
        assert calls["modal_sandbox_kwargs"]["memory"] == 4096
        assert calls["persistent_filesystem"] is True
        assert calls["task_id"] == "task1"

    def test_create_unknown_type_raises(self, monkeypatch):
        monkeypatch.setattr(terminal_tool, "_get_plugin_env_provider", lambda et: None)
        with pytest.raises(ValueError, match="Unknown environment type"):
            terminal_tool._create_environment("bogus", image="", cwd="/w", timeout=30)


# ---------------------------------------------------------------------------
# ensure_task_env
# ---------------------------------------------------------------------------
def _docker_config():
    return {
        "env_type": "docker",
        "docker_image": "img",
        "cwd": "/workspace",
        "timeout": 30,
        "host_cwd": None,
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
        "ssh_persistent": True,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "modal_mode": "auto",
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,
        "docker_forward_env": [],
        "docker_env": {},
    }


class TestEnsureTaskEnv:
    def test_local_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool, "_get_env_config", lambda: {"env_type": "local"}
        )
        assert terminal_tool.ensure_task_env("task1") is None

    def test_returns_existing_active_env(self, monkeypatch):
        existing = object()
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _docker_config())
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda tid: tid)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: existing)
        assert terminal_tool.ensure_task_env("task1") is existing

    def test_creates_and_caches_env(self, monkeypatch):
        env_obj = _CleanupRecorder()
        created = []

        def fake_create(**kwargs):
            created.append(kwargs)
            return env_obj

        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _docker_config())
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda tid: tid)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: None)
        monkeypatch.setattr(terminal_tool, "_create_environment", fake_create)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

        result = terminal_tool.ensure_task_env("task1")

        assert result is env_obj
        assert terminal_tool._active_environments["task1"] is env_obj
        assert "task1" in terminal_tool._last_activity
        assert created
        assert created[0]["env_type"] == "docker"
        assert created[0]["task_id"] == "task1"
        assert created[0]["image"] == "img"
        assert created[0]["host_cwd"] is None

    def test_creation_failure_returns_none(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("sandbox unavailable")

        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _docker_config())
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda tid: tid)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: None)
        monkeypatch.setattr(terminal_tool, "_create_environment", boom)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

        assert terminal_tool.ensure_task_env("task1") is None
        assert terminal_tool._active_environments == {}
        assert terminal_tool._last_activity == {}


# ---------------------------------------------------------------------------
# is_persistent_env
# ---------------------------------------------------------------------------
class TestIsPersistentEnv:
    def test_no_env_false(self, monkeypatch):
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: None)
        assert terminal_tool.is_persistent_env("t") is False

    def test_session_scoped_true(self, monkeypatch):
        env = types.SimpleNamespace(_session_scoped=True, _persistent=False)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: env)
        assert terminal_tool.is_persistent_env("t") is True

    def test_persistent_attr_true(self, monkeypatch):
        env = types.SimpleNamespace(_session_scoped=False, _persistent=True)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: env)
        assert terminal_tool.is_persistent_env("t") is True

    def test_not_persistent_false(self, monkeypatch):
        env = types.SimpleNamespace(_session_scoped=False, _persistent=False)
        monkeypatch.setattr(terminal_tool, "get_active_env", lambda tid: env)
        assert terminal_tool.is_persistent_env("t") is False


# ---------------------------------------------------------------------------
# cleanup_vm
# ---------------------------------------------------------------------------
class TestCleanupVm:
    def test_cleanup_nonexistent_task_noop(self, monkeypatch):
        terminal_tool.cleanup_vm("missing")  # must not raise
        assert terminal_tool._active_environments == {}

    def test_cleanup_calls_env_cleanup(self, monkeypatch):
        env = _CleanupRecorder()
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        # stub the file_ops cache invalidator so the test is hermetic
        _stub_file_tools(monkeypatch)

        terminal_tool.cleanup_vm("task1")

        assert env.cleaned is True
        assert "task1" not in terminal_tool._active_environments
        assert "task1" not in terminal_tool._last_activity

    def test_cleanup_force_remove_dispatch(self, monkeypatch):
        # DockerEnvironment-style cleanup that accepts force_remove by name.
        calls = {}

        class _DockerLike:
            def cleanup(self, force_remove=False):
                calls["force_remove"] = force_remove

        env = _DockerLike()
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        _stub_file_tools(monkeypatch)

        terminal_tool.cleanup_vm("task1", force_remove=True)
        assert calls["force_remove"] is True

    def test_cleanup_not_forced_when_env_rejects(self, monkeypatch):
        # cleanup() that does not accept a keyword -> called with no args.
        env = _CleanupRecorder(accept_force_remove=False)
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        _stub_file_tools(monkeypatch)

        terminal_tool.cleanup_vm("task1", force_remove=True)
        assert env.cleaned is True
        assert env.force_remove is None

    def test_cleanup_swallows_404_error(self, monkeypatch):
        env = _CleanupRecorder()

        def boom():
            raise ValueError("container 404 not found")

        env.cleanup = boom
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        _stub_file_tools(monkeypatch)

        terminal_tool.cleanup_vm("task1")  # must not raise
        assert "task1" not in terminal_tool._active_environments

    def test_cleanup_stop_fallback(self, monkeypatch):
        # An env with no cleanup() but a stop() method still gets torn down.
        class _StopOnly:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        env = _StopOnly()
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        _stub_file_tools(monkeypatch)

        terminal_tool.cleanup_vm("task1")
        assert env.stopped is True


# ---------------------------------------------------------------------------
# cleanup_all_environments
# ---------------------------------------------------------------------------
class TestCleanupAllEnvironments:
    def test_cleans_all(self, monkeypatch, tmp_path):
        e1 = _CleanupRecorder()
        e2 = _CleanupRecorder()
        terminal_tool._active_environments.update({"a": e1, "b": e2})
        terminal_tool._last_activity.update({"a": time.time(), "b": time.time()})
        _stub_file_tools(monkeypatch)
        monkeypatch.setattr(terminal_tool, "_get_scratch_dir", lambda: tmp_path)

        cleaned = terminal_tool.cleanup_all_environments()

        assert cleaned == 2
        assert e1.cleaned is True
        assert e2.cleaned is True
        assert terminal_tool._active_environments == {}

    def test_no_active_envs_returns_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(terminal_tool, "_get_scratch_dir", lambda: tmp_path)
        assert terminal_tool.cleanup_all_environments() == 0

    def test_continues_past_one_bad_env(self, monkeypatch, tmp_path):
        good = _CleanupRecorder()

        class _Bad:
            def cleanup(self):
                raise RuntimeError("boom")

        terminal_tool._active_environments.update({"bad": _Bad(), "good": good})
        terminal_tool._last_activity.update({"bad": time.time(), "good": time.time()})
        _stub_file_tools(monkeypatch)
        monkeypatch.setattr(terminal_tool, "_get_scratch_dir", lambda: tmp_path)

        cleaned = terminal_tool.cleanup_all_environments()
        assert cleaned == 2
        assert good.cleaned is True


# ---------------------------------------------------------------------------
# _cleanup_inactive_envs
# ---------------------------------------------------------------------------
class TestCleanupInactiveEnvs:
    def test_removes_inactive_keeps_fresh(self, monkeypatch):
        stale_env = _CleanupRecorder()
        fresh_env = _CleanupRecorder()
        now = time.time()
        terminal_tool._active_environments.update(
            {"stale": stale_env, "fresh": fresh_env}
        )
        terminal_tool._last_activity.update(
            {"stale": now - 1000, "fresh": now}
        )
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)

        assert stale_env.cleaned is True
        assert fresh_env.cleaned is False
        assert "stale" not in terminal_tool._active_environments
        assert "stale" not in terminal_tool._last_activity
        assert "fresh" in terminal_tool._active_environments
        assert "fresh" in terminal_tool._last_activity

    def test_nothing_inactive_is_kept(self, monkeypatch):
        env = _CleanupRecorder()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)
        assert env.cleaned is False

    def test_active_process_keeps_sandbox_alive(self, monkeypatch):
        env = _CleanupRecorder()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        _stub_file_tools(monkeypatch)
        # Register "t" as having active background processes.
        fake = types.ModuleType("tools.process_registry")

        class _Registry:
            @staticmethod
            def has_active_processes(task_id):
                return task_id == "t"

        fake.process_registry = _Registry()
        monkeypatch.setitem(sys.modules, "tools.process_registry", fake)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)
        assert env.cleaned is False
        assert "t" in terminal_tool._active_environments


# ---------------------------------------------------------------------------
# _atexit_cleanup
# ---------------------------------------------------------------------------
class TestAtexitCleanup:
    def test_stops_thread_and_cleans(self, monkeypatch):
        env = _CleanupRecorder()
        terminal_tool._active_environments["t"] = env
        calls = {"cleanup": 0}
        monkeypatch.setattr(terminal_tool, "_stop_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool,
            "cleanup_all_environments",
            lambda: calls.__setitem__("cleanup", 1) or 1,
        )
        terminal_tool._atexit_cleanup()
        assert calls["cleanup"] == 1

    def test_no_envs_is_noop(self, monkeypatch):
        monkeypatch.setattr(terminal_tool, "_stop_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal_tool, "cleanup_all_environments", lambda: 0)
        terminal_tool._atexit_cleanup()  # must not raise


# ---------------------------------------------------------------------------
# _evict_environment_for_task
# ---------------------------------------------------------------------------
class TestEvictEnvironmentForTask:
    def test_evicts_and_cleans(self, monkeypatch):
        env = _CleanupRecorder()
        terminal_tool._active_environments["task1"] = env
        terminal_tool._last_activity["task1"] = time.time()
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda tid: tid)
        _stub_file_tools(monkeypatch)

        terminal_tool._evict_environment_for_task("task1")

        assert env.cleaned is True
        assert "task1" not in terminal_tool._active_environments
        assert "task1" not in terminal_tool._last_activity

    def test_evict_swallows_cleanup_error(self, monkeypatch):
        env = _CleanupRecorder()
        terminal_tool._active_environments["task1"] = env
        monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda tid: tid)
        monkeypatch.setattr(env, "cleanup", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        terminal_tool._evict_environment_for_task("task1")  # must not raise
        assert "task1" not in terminal_tool._active_environments


# ---------------------------------------------------------------------------
# Task env overrides
# ---------------------------------------------------------------------------
class TestTaskEnvOverrides:
    def test_register_stores_overrides(self, monkeypatch):
        terminal_tool.register_task_env_overrides("t1", {"docker_image": "img"})
        assert terminal_tool._task_env_overrides["t1"] == {"docker_image": "img"}

    def test_register_cwd_updates_env(self, monkeypatch):
        env = types.SimpleNamespace(cwd="/old")
        terminal_tool._active_environments["t1"] = env
        monkeypatch.setattr(
            terminal_tool, "_resolve_container_task_id", lambda tid: tid
        )
        terminal_tool.register_task_env_overrides("t1", {"cwd": "/proj/root"})
        assert env.cwd == "/proj/root"
        assert terminal_tool._session_cwd["t1"] == "/proj/root"

    def test_clear_drops_all_artifacts(self, monkeypatch):
        terminal_tool._task_env_overrides["t1"] = {"x": 1}
        terminal_tool._session_cwd["t1"] = "/proj"
        terminal_tool._container_aliases["t1"] = "default"
        terminal_tool.clear_task_env_overrides("t1")
        assert terminal_tool._task_env_overrides == {}
        assert terminal_tool._session_cwd == {}
        assert terminal_tool._container_aliases == {}


# ---------------------------------------------------------------------------
# Container aliasing
# ---------------------------------------------------------------------------
class TestContainerAlias:
    def test_register_alias(self, monkeypatch):
        terminal_tool.register_container_alias("child", "parent")
        assert terminal_tool._container_aliases["child"] == "parent"

    def test_register_alias_none_parent_defaults(self, monkeypatch):
        terminal_tool.register_container_alias("child", None)
        assert terminal_tool._container_aliases["child"] == "default"

    def test_register_alias_empty_child_noop(self, monkeypatch):
        terminal_tool.register_container_alias("", "parent")
        assert terminal_tool._container_aliases == {}

    def test_resolve_alias_chain(self, monkeypatch):
        terminal_tool._container_aliases.update({"child": "mid", "mid": "default"})
        assert terminal_tool._resolve_container_alias("child") == "default"

    def test_resolve_alias_no_entry_returns_self(self, monkeypatch):
        assert terminal_tool._resolve_container_alias("standalone") == "standalone"

    def test_resolve_alias_cycle_safe(self, monkeypatch):
        terminal_tool._container_aliases.update({"a": "b", "b": "a"})
        # Must terminate (no infinite loop); returns the last-hop key.
        assert terminal_tool._resolve_container_alias("a") in {"a", "b"}


# ---------------------------------------------------------------------------
# fixture-internal helpers (defined at bottom to keep fixtures coherent)
# ---------------------------------------------------------------------------
def _stub_file_tools(monkeypatch):
    """Neutralise the file_ops cache invalidation in cleanup paths."""
    fake = types.ModuleType("tools.file_tools")
    fake.clear_file_ops_cache = lambda task_id: None
    monkeypatch.setitem(sys.modules, "tools.file_tools", fake)


def _capture_logger(info=None, warning=None, debug=None, error=None):
    """A logger double that appends (args, kwargs) to the requested lists."""

    class _Logger:
        def info(self, *a, **k):
            if info is not None:
                info.append((a, k))

        def warning(self, *a, **k):
            if warning is not None:
                warning.append((a, k))

        def debug(self, *a, **k):
            if debug is not None:
                debug.append((a, k))

        def error(self, *a, **k):
            if error is not None:
                error.append((a, k))

    return _Logger()


# ---------------------------------------------------------------------------
# _maybe_reap_docker_orphans
# ---------------------------------------------------------------------------
class TestMaybeReapDockerOrphans:
    def _reset(self, monkeypatch):
        # The reaper is a once-per-process guard; reset it (and its lock) so
        # each test can observe a fresh sweep independent of earlier cases.
        monkeypatch.setattr(terminal_tool, "_docker_orphan_reaper_ran", False)
        monkeypatch.setattr(
            terminal_tool, "_docker_orphan_reaper_lock", threading.Lock()
        )

    def _fake_docker_module(self, monkeypatch, reap_returns=0, reap_raises=None):
        """Install a fake tools.environments.docker with recorded reap calls."""
        fake = types.ModuleType("tools.environments.docker")
        fake._container_identity = lambda key: f"profile:{key}"
        calls = {}

        if reap_raises is not None:

            def boom(max_age_seconds, profile_filter):
                raise reap_raises

            fake.reap_orphan_containers = boom
        else:

            def reap(max_age_seconds, profile_filter):
                calls["max_age"] = max_age_seconds
                calls["profile"] = profile_filter
                return reap_returns

            fake.reap_orphan_containers = reap

        fake._reap_calls = calls
        monkeypatch.setitem(sys.modules, "tools.environments.docker", fake)
        return fake

    def test_disabled_returns_early(self, monkeypatch):
        self._reset(monkeypatch)
        terminal_tool._maybe_reap_docker_orphans({"docker_orphan_reaper": False})
        # Early return: the once-per-process guard must NOT have been set.
        assert terminal_tool._docker_orphan_reaper_ran is False

    def test_already_ran_skips_sweep(self, monkeypatch):
        self._reset(monkeypatch)
        terminal_tool._docker_orphan_reaper_ran = True
        terminal_tool._maybe_reap_docker_orphans({})
        assert terminal_tool._docker_orphan_reaper_ran is True

    def test_sweeps_and_logs_removed(self, monkeypatch):
        self._reset(monkeypatch)
        info = []
        monkeypatch.setattr(terminal_tool, "logger", _capture_logger(info=info))
        fake = self._fake_docker_module(monkeypatch, reap_returns=2)
        # Hermetic fixture already drops TERMINAL_* => default lifetime 300.
        monkeypatch.delenv("TERMINAL_LIFETIME_SECONDS", raising=False)

        terminal_tool._maybe_reap_docker_orphans(
            {"docker_shared_container_key": "shared:key"}
        )

        assert fake._reap_calls["max_age"] == 600  # 2 x lifetime_seconds
        assert fake._reap_calls["profile"] == "profile:shared:key"
        # logger.info(msg, removed, profile) — logging defers %-interpolation.
        assert info and info[0][0][1] == 2
        assert info[0][0][2] == "profile:shared:key"

    def test_invalid_lifetime_falls_back_to_default(self, monkeypatch):
        self._reset(monkeypatch)
        monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "not-an-int")
        fake = self._fake_docker_module(monkeypatch, reap_returns=0)

        terminal_tool._maybe_reap_docker_orphans({})

        assert fake._reap_calls["max_age"] == 600  # default 300 * 2

    def test_lifetime_floor_at_60(self, monkeypatch):
        self._reset(monkeypatch)
        monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "0")
        fake = self._fake_docker_module(monkeypatch, reap_returns=0)

        terminal_tool._maybe_reap_docker_orphans({})

        # max(60, 0) * 2 == 120, not an instant reap that races our own setup.
        assert fake._reap_calls["max_age"] == 120

    def test_import_error_returns_gracefully(self, monkeypatch):
        self._reset(monkeypatch)
        # Empty module lacks reap_orphan_containers/_container_identity =>
        # the lazy import inside the function raises and we bail out.
        monkeypatch.setitem(
            sys.modules, "tools.environments.docker", types.ModuleType("tools.environments.docker")
        )
        terminal_tool._maybe_reap_docker_orphans({})  # must not raise
        assert terminal_tool._docker_orphan_reaper_ran is True

    def test_reap_exception_swallowed(self, monkeypatch):
        self._reset(monkeypatch)
        debug = []
        monkeypatch.setattr(terminal_tool, "logger", _capture_logger(debug=debug))
        self._fake_docker_module(
            monkeypatch, reap_raises=RuntimeError("docker daemon down")
        )
        terminal_tool._maybe_reap_docker_orphans({})  # must not raise
        assert terminal_tool._docker_orphan_reaper_ran is True
        assert debug  # debug log recorded the janitor failure


# ---------------------------------------------------------------------------
# Cleanup background thread: _start / _stop / _cleanup_thread_worker
# ---------------------------------------------------------------------------
class TestCleanupThread:
    def _stop_thread(self):
        terminal_tool._cleanup_running = False
        t = terminal_tool._cleanup_thread
        if t is not None and t.is_alive():
            t.join(timeout=2)
        terminal_tool._cleanup_thread = None

    def test_start_creates_daemon_thread(self, monkeypatch):
        self._stop_thread()
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"lifetime_seconds": 1})
        monkeypatch.setattr(terminal_tool, "_cleanup_inactive_envs", lambda ls: None)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        terminal_tool._start_cleanup_thread()

        assert terminal_tool._cleanup_running is True
        assert terminal_tool._cleanup_thread is not None
        assert terminal_tool._cleanup_thread.is_alive()
        assert terminal_tool._cleanup_thread.daemon is True
        self._stop_thread()

    def test_start_is_idempotent(self, monkeypatch):
        self._stop_thread()
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"lifetime_seconds": 1})
        monkeypatch.setattr(terminal_tool, "_cleanup_inactive_envs", lambda ls: None)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        terminal_tool._start_cleanup_thread()
        first = terminal_tool._cleanup_thread
        terminal_tool._start_cleanup_thread()

        assert terminal_tool._cleanup_thread is first
        self._stop_thread()

    def test_stop_stops_thread(self, monkeypatch):
        self._stop_thread()
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"lifetime_seconds": 1})
        monkeypatch.setattr(terminal_tool, "_cleanup_inactive_envs", lambda ls: None)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        terminal_tool._start_cleanup_thread()
        thread = terminal_tool._cleanup_thread
        terminal_tool._stop_cleanup_thread()

        assert terminal_tool._cleanup_running is False
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_stop_with_no_thread_is_noop(self, monkeypatch):
        self._stop_thread()
        monkeypatch.setattr(terminal_tool, "_cleanup_thread", None)
        terminal_tool._stop_cleanup_thread()
        assert terminal_tool._cleanup_running is False

    def test_worker_runs_cleanup_with_lifetime(self, monkeypatch):
        self._stop_thread()
        calls = []
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"lifetime_seconds": 42})
        monkeypatch.setattr(terminal_tool, "_cleanup_inactive_envs", lambda ls: calls.append(ls))
        real_sleep = time.sleep
        ran_once = threading.Event()

        def fake_sleep(secs):
            if not ran_once.is_set():
                ran_once.set()
            real_sleep(0.005)

        monkeypatch.setattr(time, "sleep", fake_sleep)
        terminal_tool._cleanup_running = True
        t = threading.Thread(target=terminal_tool._cleanup_thread_worker, daemon=True)
        t.start()

        assert ran_once.wait(2)
        terminal_tool._cleanup_running = False
        t.join(timeout=2)

        assert not t.is_alive()
        assert calls and calls[0] == 42

    def test_worker_swallows_error_logs_warning(self, monkeypatch):
        self._stop_thread()
        warnings = []
        monkeypatch.setattr(terminal_tool, "logger", _capture_logger(warning=warnings))

        def boom():
            raise RuntimeError("config read failed")

        monkeypatch.setattr(terminal_tool, "_get_env_config", boom)
        real_sleep = time.sleep
        ran_once = threading.Event()

        def fake_sleep(secs):
            if not ran_once.is_set():
                ran_once.set()
            real_sleep(0.005)

        monkeypatch.setattr(time, "sleep", fake_sleep)
        terminal_tool._cleanup_running = True
        t = threading.Thread(target=terminal_tool._cleanup_thread_worker, daemon=True)
        t.start()

        assert ran_once.wait(2)
        terminal_tool._cleanup_running = False
        t.join(timeout=2)

        assert not t.is_alive()
        assert warnings


# ---------------------------------------------------------------------------
# _cleanup_inactive_envs — remaining branches
# ---------------------------------------------------------------------------
class TestCleanupInactiveEnvsEdges:
    def test_process_registry_import_error_handled(self, monkeypatch):
        # Force `from tools.process_registry import process_registry` to fail so
        # the whole sandbox-keepalive block is skipped without crashing.
        monkeypatch.setitem(
            sys.modules, "tools.process_registry", types.ModuleType("tools.process_registry")
        )
        env = _CleanupRecorder()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)

        assert env.cleaned is True

    def test_file_tools_import_error_handled(self, monkeypatch):
        # Force the file_ops cache invalidation import to fail.
        _fake_process_registry(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tools.file_tools", types.ModuleType("tools.file_tools")
        )
        env = _CleanupRecorder()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)

        assert env.cleaned is True

    def test_cleanup_stop_fallback(self, monkeypatch):
        class _StopOnly:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        env = _StopOnly()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)

        assert env.stopped is True

    def test_cleanup_terminate_fallback(self, monkeypatch):
        class _TerminateOnly:
            def __init__(self):
                self.terminated = False

            def terminate(self):
                self.terminated = True

        env = _TerminateOnly()
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)

        assert env.terminated is True

    def test_404_error_logged_as_already_cleaned(self, monkeypatch):
        def boom():
            raise ValueError("container 404 not found")

        env = _CleanupRecorder()
        env.cleanup = boom
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        info = []
        monkeypatch.setattr(terminal_tool, "logger", _capture_logger(info=info))
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)  # must not raise

        assert info and "already cleaned up" in str(info[0])

    def test_other_cleanup_error_logged_warning(self, monkeypatch):
        def boom():
            raise RuntimeError("disk full")

        env = _CleanupRecorder()
        env.cleanup = boom
        now = time.time()
        terminal_tool._active_environments["t"] = env
        terminal_tool._last_activity["t"] = now - 1000
        warning = []
        monkeypatch.setattr(terminal_tool, "logger", _capture_logger(warning=warning))
        _fake_process_registry(monkeypatch)
        _stub_file_tools(monkeypatch)

        terminal_tool._cleanup_inactive_envs(lifetime_seconds=300)  # must not raise

        assert warning


# ---------------------------------------------------------------------------
# _rewrite_compound_background — branch coverage (parens / semicolon / pipe /
# comment without trailing newline)
# ---------------------------------------------------------------------------
class TestRewriteCompoundBackgroundBranches:
    def test_paren_group_untouched(self):
        assert (
            terminal_tool._rewrite_compound_background("(A && B &)") == "(A && B &)"
        )

    def test_semicolon_resets_chain(self):
        cmd = "A && B ; C &"
        assert terminal_tool._rewrite_compound_background(cmd) == cmd

    def test_pipe_resets_chain(self):
        cmd = "A && B | C &"
        assert terminal_tool._rewrite_compound_background(cmd) == cmd

    def test_comment_no_trailing_newline_untouched(self):
        cmd = "echo hi # A && B &"
        assert terminal_tool._rewrite_compound_background(cmd) == cmd

    def test_comment_only_no_newline_untouched(self):
        cmd = "# just a comment"
        assert terminal_tool._rewrite_compound_background(cmd) == cmd
