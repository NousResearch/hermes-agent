"""Behavior tests for standalone terminal-backend plugins."""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.environments.manager import environment_manager
from tools.environments.registry import (
    TerminalBackendDefinition,
    TerminalBackendRequest,
    terminal_backend_registry,
)


class _FixtureEnvironment:
    def __init__(self, request: TerminalBackendRequest):
        self.request = request
        self.cwd = request.cwd
        self.timeout = request.timeout
        self.cleanup_calls = 0

    def execute(self, command: str, cwd: str = "", **_kwargs):
        if "kernel=%s" in command:
            return {
                "output": "os=Linux\nkernel=fixture\nhome=/root\ncwd=/workspace\nuser=fixture",
                "returncode": 0,
            }
        return {
            "output": f"fixture:{command}",
            "returncode": 0,
            "cwd": cwd or self.cwd,
        }

    def cleanup(self):
        self.cleanup_calls += 1


def _reset_environment_state():
    """Empty the environment caches and re-bind the manager to the live ones.

    _get_or_create_environment re-points the manager at terminal_tool's
    module-level lifecycle dicts on every call, for tests that patch those
    dicts. When such a test restores its patch the manager still holds the
    patched copy, so clearing only one side leaves stale entries behind.
    """
    import tools.terminal_tool as terminal_tool

    with environment_manager.lock:
        environment_manager.active_environments.clear()
        environment_manager.last_activity.clear()
        terminal_tool._active_environments.clear()
        terminal_tool._last_activity.clear()
        environment_manager.active_environments = terminal_tool._active_environments
        environment_manager.last_activity = terminal_tool._last_activity
    with environment_manager.creation_locks_lock:
        environment_manager.creation_locks.clear()
        environment_manager._creation_lock_users.clear()
        terminal_tool._creation_locks.clear()
        environment_manager.creation_locks = terminal_tool._creation_locks
    environment_manager.lock = terminal_tool._env_lock
    environment_manager.creation_locks_lock = terminal_tool._creation_locks_lock


@pytest.fixture(autouse=True)
def _clean_terminal_backend_state(monkeypatch):
    import tools.terminal_tool as terminal_tool

    terminal_backend_registry.unregister_plugin_backends()
    _reset_environment_state()
    # Keep the background cleanup daemon out of every test in this module.
    # _get_or_create_environment starts it on the first environment it builds,
    # and it then evicts entries from the very caches these tests assert on,
    # on its own schedule, for the rest of the process.
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    yield
    terminal_backend_registry.unregister_plugin_backends()
    _reset_environment_state()
    try:
        from tools.file_tools import clear_file_ops_cache

        clear_file_ops_cache()
    except ImportError:
        pass


def _context(name: str) -> PluginContext:
    return PluginContext(
        PluginManifest(name=name, key=name, source="entrypoint"),
        PluginManager(),
    )


def test_definition_is_frozen_and_registry_keeps_host_copy():
    definition = TerminalBackendDefinition(
        name="fixture",
        factory=_FixtureEnvironment,
        default_cwd="/workspace",
    )
    _context("fixture-plugin").register_terminal_backend(definition)

    with pytest.raises(FrozenInstanceError):
        definition.default_cwd = "/changed"  # type: ignore[misc]

    hosted = terminal_backend_registry.get("fixture")
    assert hosted is not definition
    assert hosted == definition


def test_registration_rejects_reserved_and_cross_plugin_duplicate_names():
    with pytest.raises(ValueError, match="reserved"):
        TerminalBackendDefinition(name="docker", factory=_FixtureEnvironment)

    definition = TerminalBackendDefinition(name="fixture", factory=_FixtureEnvironment)
    _context("first-plugin").register_terminal_backend(definition)
    with pytest.raises(ValueError, match="already registered"):
        _context("second-plugin").register_terminal_backend(definition)


def test_failed_plugin_register_rolls_back_owned_terminal_backends(monkeypatch):
    manager = PluginManager()
    stable_context = PluginContext(
        PluginManifest(name="stable", key="stable", source="entrypoint"),
        manager,
    )
    stable_context.register_terminal_backend(
        TerminalBackendDefinition(name="stable", factory=_FixtureEnvironment)
    )

    def register(ctx):
        ctx.register_terminal_backend(
            TerminalBackendDefinition(name="partial", factory=_FixtureEnvironment)
        )
        raise RuntimeError("register failed")

    manifest = PluginManifest(name="broken", key="broken", source="entrypoint")
    monkeypatch.setattr(
        manager,
        "_load_entrypoint_module",
        lambda _manifest: SimpleNamespace(register=register),
    )

    manager._load_plugin(manifest)

    assert terminal_backend_registry.get("partial") is None
    assert terminal_backend_registry.get("stable") is not None
    assert manager._plugin_terminal_backend_names == {"stable": {"stable"}}
    assert manager._plugins["broken"].error == "register failed"


def test_failed_discovery_sweep_restores_terminal_backend_snapshot(monkeypatch):
    manager = PluginManager()
    stable_context = PluginContext(
        PluginManifest(name="stable", key="stable", source="entrypoint"),
        manager,
    )
    stable_context.register_terminal_backend(
        TerminalBackendDefinition(name="stable", factory=_FixtureEnvironment)
    )

    def fail_sweep():
        partial_context = PluginContext(
            PluginManifest(name="partial", key="partial", source="entrypoint"),
            manager,
        )
        partial_context.register_terminal_backend(
            TerminalBackendDefinition(name="partial", factory=_FixtureEnvironment)
        )
        raise RuntimeError("sweep failed")

    monkeypatch.setattr(manager, "_discover_and_load_inner", fail_sweep)

    with pytest.raises(RuntimeError, match="sweep failed"):
        manager.discover_and_load()

    assert manager._discovered is False
    assert terminal_backend_registry.get("partial") is None
    assert terminal_backend_registry.get("stable") is not None
    assert manager._plugin_terminal_backend_names == {"stable": {"stable"}}


def test_terminal_file_and_code_tools_share_one_plugin_environment(monkeypatch):
    requests: list[TerminalBackendRequest] = []

    def factory(request: TerminalBackendRequest):
        requests.append(request)
        return _FixtureEnvironment(request)

    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(
            name="fixture",
            factory=factory,
            container_paths=True,
            default_cwd="/workspace",
            default_image="fixture/default:latest",
            image_override_key="fixture_image",
            image_config_key="image",
        )
    )
    monkeypatch.setenv("TERMINAL_ENV", "fixture")

    from tools.code_execution_tool import _get_or_create_env
    from tools.file_tools import _get_file_ops, _uses_container_paths
    import tools.terminal_tool as terminal_module

    plugin_settings = {"image": "fixture/test:1", "nested": {"items": []}}
    monkeypatch.setattr(
        terminal_module,
        "_get_plugin_backend_settings",
        lambda _env_type: plugin_settings,
    )

    terminal_env, env_type, created = terminal_module._get_or_create_environment(
        "shared-task"
    )
    monkeypatch.setattr(
        terminal_module,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    terminal_result = json.loads(
        terminal_module.terminal_tool("echo routed", task_id="shared-task")
    )
    file_ops = _get_file_ops("shared-task")
    code_env, code_env_type = _get_or_create_env("shared-task")

    assert created is True
    assert terminal_result["exit_code"] == 0
    assert terminal_result["output"] == "fixture:echo routed"
    assert env_type == code_env_type == "fixture"
    assert code_env is terminal_env
    assert file_ops.env is terminal_env
    assert _uses_container_paths("shared-task") is True
    assert len(requests) == 1
    assert requests[0].cwd == "/workspace"
    assert requests[0].image == "fixture/test:1"
    assert requests[0].settings["image"] == "fixture/test:1"
    with pytest.raises(TypeError):
        requests[0].settings["image"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        requests[0].settings["nested"]["items"] = ("plugin-change",)  # type: ignore[index]
    with pytest.raises(AttributeError):
        requests[0].settings["nested"]["items"].append("plugin-change")
    assert plugin_settings["nested"]["items"] == []


def test_cached_file_operations_refresh_environment_activity(monkeypatch):
    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="fixture", factory=_FixtureEnvironment)
    )
    monkeypatch.setenv("TERMINAL_ENV", "fixture")

    import tools.environments.manager as manager_module
    import tools.terminal_tool as terminal_module
    from tools.file_tools import _get_file_ops

    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    file_ops = _get_file_ops("file-task")
    with environment_manager.lock:
        environment_manager.last_activity["default"] = 0.0
    monkeypatch.setattr(manager_module.time, "time", lambda: 100.0)

    assert _get_file_ops("file-task") is file_ops
    assert environment_manager.last_activity["default"] == 100.0

    monkeypatch.setattr(terminal_module.time, "time", lambda: 100.0)
    terminal_module._cleanup_inactive_envs(lifetime_seconds=50)
    assert environment_manager.active_environments["default"] is file_ops.env


def test_invalid_factory_result_is_not_cached(monkeypatch):
    _context("broken-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="broken", factory=lambda _request: object())
    )
    monkeypatch.setenv("TERMINAL_ENV", "broken")

    from tools.terminal_tool import _get_or_create_environment

    with pytest.raises(TypeError, match="missing callable methods"):
        _get_or_create_environment("broken-task")
    assert environment_manager.active_environments == {}
    assert environment_manager.creation_locks == {}


def test_public_terminal_and_file_definitions_accept_registered_backend(monkeypatch):
    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="fixture", factory=_FixtureEnvironment)
    )
    import tools.terminal_tool as terminal_module
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {"env_type": "fixture"},
    )
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()

    definitions = get_tool_definitions(
        enabled_toolsets=["terminal", "file"],
        quiet_mode=True,
    )
    names = {definition["function"]["name"] for definition in definitions}

    assert {"terminal", "read_file", "write_file", "patch", "search_files"} <= names


def test_prompt_probe_uses_the_managed_plugin_environment(monkeypatch):
    requests: list[TerminalBackendRequest] = []

    def factory(request):
        requests.append(request)
        return _FixtureEnvironment(request)

    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="fixture", factory=factory)
    )
    import tools.terminal_tool as terminal_module
    from agent import prompt_builder

    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {
            "env_type": "fixture",
            "cwd": "/workspace",
            "timeout": 30,
            "host_cwd": None,
        },
    )
    monkeypatch.setattr(
        terminal_module,
        "_get_plugin_backend_settings",
        lambda _env_type: {},
    )
    monkeypatch.setenv("TERMINAL_ENV", "fixture")
    prompt_builder._BACKEND_PROBE_CACHE.clear()

    summary = prompt_builder._probe_remote_backend("fixture")
    hints = prompt_builder.build_environment_hints()
    terminal_env, _env_type, created = terminal_module._get_or_create_environment(
        "default"
    )

    assert "OS: Linux fixture" in summary
    assert "Terminal backend: fixture" in hints
    assert "NOT on the machine where Hermes itself is running" in hints
    assert terminal_env is environment_manager.active_environments["default"]
    assert created is False
    assert len(requests) == 1


def test_cleanup_waits_for_creation_and_cleans_the_created_environment(monkeypatch):
    factory_started = threading.Event()
    release_factory = threading.Event()
    created: list[_FixtureEnvironment] = []

    def factory(request):
        factory_started.set()
        release_factory.wait(timeout=2)
        env = _FixtureEnvironment(request)
        created.append(env)
        return env

    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="fixture", factory=factory)
    )
    monkeypatch.setenv("TERMINAL_ENV", "fixture")
    import tools.terminal_tool as terminal_module

    create_thread = threading.Thread(
        target=terminal_module._get_or_create_environment,
        args=("race-task",),
    )
    create_thread.start()
    assert factory_started.wait(timeout=1)
    cleanup_thread = threading.Thread(
        target=terminal_module.cleanup_vm,
        args=("race-task",),
    )
    cleanup_thread.start()
    release_factory.set()
    create_thread.join(timeout=2)
    cleanup_thread.join(timeout=2)

    assert not create_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert len(created) == 1
    assert created[0].cleanup_calls == 1
    assert environment_manager.active_environments == {}
    assert environment_manager.creation_locks == {}


def test_cleanup_supports_uninspectable_backend_callable():
    import tools.terminal_tool as terminal_module

    calls = []

    class OpaqueCleanup:
        @property
        def __signature__(self):
            raise ValueError("opaque SDK callable")

        def __call__(self):
            calls.append("cleanup")

    terminal_module._invoke_environment_cleanup(
        type("OpaqueEnvironment", (), {"cleanup": OpaqueCleanup()})()
    )

    assert calls == ["cleanup"]


def test_cleanup_absence_requires_structured_404():
    import tools.terminal_tool as terminal_module

    class FailingEnvironment:
        error: Exception

        def cleanup(self):
            raise self.error

    env = FailingEnvironment()
    env.error = RuntimeError("resource not found")
    with pytest.raises(RuntimeError, match="not found"):
        terminal_module._cleanup_environment_absent_ok(env)

    not_found = RuntimeError("control-plane response")
    not_found.status_code = 404
    env.error = RuntimeError("cleanup failed")
    env.error.__cause__ = not_found
    terminal_module._cleanup_environment_absent_ok(env)


def test_safe_mode_clears_terminal_backend_registrations(monkeypatch):
    _context("fixture-plugin").register_terminal_backend(
        TerminalBackendDefinition(name="fixture", factory=_FixtureEnvironment)
    )
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")

    PluginManager().discover_and_load(force=True)

    assert terminal_backend_registry.get("fixture") is None
