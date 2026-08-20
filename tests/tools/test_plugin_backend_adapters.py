from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tools.environments import (
    BackendCapabilities,
    BackendDefinition,
    BackendFactoryRequest,
    BaseEnvironment,
    ExecutionLocation,
    FilesystemSemantics,
)
from tools.environments.registry import terminal_backend_registry


@pytest.fixture(autouse=True)
def _reset_backend_runtime():
    terminal_backend_registry.reset()
    yield
    terminal_backend_registry.reset()


def _definition(*, available: bool = True) -> BackendDefinition:
    return BackendDefinition(
        name="third_party_remote",
        label="Third-party Remote",
        description="a third-party remote workspace",
        factory=lambda _request: MagicMock(spec=BaseEnvironment),
        availability_check=lambda: available,
        capabilities=BackendCapabilities(
            execution_location=ExecutionLocation.REMOTE,
            filesystem_semantics=FilesystemSemantics.ISOLATED,
            requires_sandbox_cwd=True,
        ),
    )


def test_registered_config_uses_sandbox_default_cwd(monkeypatch):
    from tools.terminal_tool import _get_env_config

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    monkeypatch.setenv("TERMINAL_CWD", "/opt/host-only/project")

    config = _get_env_config()

    assert config["cwd"] == "~"
    assert config["host_cwd"] is None


def test_registered_config_does_not_trust_plugin_declared_host_default(
    monkeypatch,
):
    from tools.terminal_tool import _get_env_config

    definition = _definition()
    definition.default_cwd = "/opt/host-only/project"
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = _get_env_config()

    assert config["cwd"] == "~"
    assert config["host_cwd"] is None


@pytest.mark.parametrize(
    "candidate",
    [
        "/opt/host-only/project",
        "/home/host-user/project",
        "C:\\Users\\host\\repo",
        "src",
    ],
)
def test_isolated_plugin_rejects_unverified_resolved_cwd(monkeypatch, candidate):
    from tools.terminal_tool import _sanitize_registered_sandbox_cwd

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)

    assert _sanitize_registered_sandbox_cwd(definition.name, candidate, "~") == "~"


def test_isolated_plugin_cannot_treat_caller_host_cwd_as_safe_default(monkeypatch):
    from tools.terminal_tool import _sanitize_registered_sandbox_cwd

    definition = _definition()
    definition.default_cwd = "/sandbox"
    terminal_backend_registry.register(definition)

    host_cwd = "/home/host-user/project"
    assert (
        _sanitize_registered_sandbox_cwd(
            definition.name,
            host_cwd,
            host_cwd,
        )
        == "~"
    )


def test_plugin_instance_cwd_is_not_host_verified(monkeypatch):
    from tools.terminal_tool import _sanitize_registered_sandbox_cwd

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)

    assert (
        _sanitize_registered_sandbox_cwd(
            definition.name,
            "/home/coder/project",
            "~",
            verified_cwd="/home/coder/project",
        )
        == "~"
    )


@pytest.mark.parametrize(
    ("semantics", "requires_sandbox_cwd", "accepts_host_cwd"),
    [
        (FilesystemSemantics.UNKNOWN, False, False),
        (FilesystemSemantics.HOST, False, True),
        (FilesystemSemantics.ISOLATED, False, False),
        (FilesystemSemantics.ISOLATED, True, False),
        (FilesystemSemantics.ISOLATED, True, True),
    ],
)
def test_terminal_uses_declared_filesystem_semantics_for_task_cwd(
    monkeypatch, semantics, requires_sandbox_cwd, accepts_host_cwd
):
    import tools.terminal_tool as terminal_tool

    calls = []

    class FakeEnv:
        env = {}
        cwd = "~"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    definition = _definition()
    definition.default_cwd = "~"
    definition.capabilities.filesystem_semantics = semantics
    definition.capabilities.requires_sandbox_cwd = requires_sandbox_cwd
    definition.capabilities.accepts_host_cwd = accepts_host_cwd
    terminal_backend_registry.register(definition)
    task_id = "isolated-plugin-task"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {task_id: {"cwd": "/opt/host-only/project"}},
    )
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": definition.name,
            "cwd": "~",
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    expected_cwd = (
        "/opt/host-only/project"
        if semantics is FilesystemSemantics.HOST
        or (accepts_host_cwd and semantics is not FilesystemSemantics.ISOLATED)
        else "~"
    )
    assert calls == [
        ("pwd", {"timeout": 60, "cwd": expected_cwd, "bounded_capture": True})
    ]


def test_execute_code_sanitizes_task_cwd_for_isolated_backend(monkeypatch):
    import tools.code_execution_tool as code_execution_tool
    import tools.terminal_tool as terminal_tool

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)

    task_id = "isolated-execute-code-task"
    received: dict[str, object] = {}
    environment = object()

    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(
        terminal_tool, "resolve_task_overrides", lambda _task_id: {"cwd": "/opt/host-only/project"}
    )
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": definition.name,
            "cwd": "~",
            "host_cwd": None,
            "timeout": 60,
            "backend_config": {"workspace_name": "execute-code-workspace"},
        },
    )

    def create_environment(**kwargs):
        received.update(kwargs)
        return environment

    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    resolved_environment, env_type = code_execution_tool._get_or_create_env(task_id)

    assert resolved_environment is environment
    assert env_type == definition.name
    assert received["cwd"] == "~"
    assert received["backend_config"] == {
        "workspace_name": "execute-code-workspace"
    }


def test_file_tools_sanitize_task_cwd_when_creating_isolated_backend(monkeypatch):
    import tools.file_tools as file_tools
    import tools.terminal_tool as terminal_tool

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)

    task_id = "isolated-file-tools-task"
    received: dict[str, object] = {}
    environment = MagicMock(spec=BaseEnvironment)
    environment.cwd = "~"

    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(
        terminal_tool,
        "resolve_task_overrides",
        lambda _task_id: {"cwd": "/opt/host-only/project"},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": definition.name,
            "cwd": "~",
            "host_cwd": None,
            "timeout": 60,
            "backend_config": {"workspace_name": "file-tools-workspace"},
        },
    )

    def create_environment(**kwargs):
        received.update(kwargs)
        return environment

    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    file_tools._get_file_ops(task_id)

    assert received["cwd"] == "~"
    assert received["backend_config"] == {
        "workspace_name": "file-tools-workspace"
    }


def test_background_terminal_passes_sanitized_cwd_to_plugin_local_environment(
    monkeypatch,
):
    import tools.terminal_tool as terminal_tool
    import tools.process_registry as process_registry_module
    from tools.environments.local import LocalEnvironment

    calls = []

    class CapturingLocalEnvironment(LocalEnvironment):
        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "4321\n", "returncode": 0}

    definition = _definition()
    definition.default_cwd = "~"
    definition.factory = lambda _request: CapturingLocalEnvironment(
        cwd="/opt/host-only/project"
    )
    terminal_backend_registry.register(definition)
    task_id = "isolated-plugin-background-task"
    env = CapturingLocalEnvironment(cwd="/opt/host-only/project")
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {task_id: {"cwd": "/opt/host-only/project"}},
    )
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": definition.name,
            "cwd": "~",
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    fake_thread = MagicMock()
    monkeypatch.setattr(
        process_registry_module.threading, "Thread", lambda **_kwargs: fake_thread
    )
    monkeypatch.setattr(
        process_registry_module.process_registry,
        "_write_checkpoint",
        lambda: None,
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            background=True,
            notify_on_complete=False,
        )
    )

    try:
        assert result["exit_code"] == 0
        assert calls[0][1]["cwd"] == "~"
        assert "/opt/host-only/project" not in calls[0][1].values()
    finally:
        process_registry_module.process_registry._running.pop(
            result["session_id"], None
        )


def test_requirements_accept_registered_available_backend(monkeypatch):
    from tools.terminal_tool import check_terminal_requirements

    terminal_backend_registry.register(_definition())
    monkeypatch.setenv("TERMINAL_ENV", "third_party_remote")

    assert check_terminal_requirements() is True


def test_requirements_reject_registered_unavailable_backend(
    monkeypatch, caplog
):
    from tools.terminal_tool import check_terminal_requirements

    terminal_backend_registry.register(_definition(available=False))
    monkeypatch.setenv("TERMINAL_ENV", "third_party_remote")

    assert check_terminal_requirements() is False
    assert (
        "Terminal backend 'third_party_remote' is unavailable"
        in caplog.text
    )


def test_remote_capability_suppresses_host_prompt(monkeypatch):
    import agent.prompt_builder as prompt_builder

    terminal_backend_registry.register(_definition())
    monkeypatch.setenv("TERMINAL_ENV", "third_party_remote")
    monkeypatch.setattr(prompt_builder, "is_wsl", lambda: False)
    monkeypatch.setattr(prompt_builder, "_probe_remote_backend", lambda _name: None)
    prompt_builder._clear_backend_probe_cache()

    hint = prompt_builder.build_environment_hints()

    assert "Host:" not in hint
    assert "Current working directory:" not in hint
    assert "Terminal backend: third_party_remote" in hint
    assert "third-party remote workspace" in hint


def test_unknown_location_suppresses_host_prompt(monkeypatch):
    import agent.prompt_builder as prompt_builder

    definition = _definition()
    definition.capabilities.execution_location = ExecutionLocation.UNKNOWN
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    monkeypatch.setattr(prompt_builder, "_probe_remote_backend", lambda _name: None)
    prompt_builder._clear_backend_probe_cache()

    hint = prompt_builder.build_environment_hints()

    assert "Host:" not in hint
    assert "Current working directory:" not in hint


def test_local_capability_exposes_host_prompt(monkeypatch):
    import agent.prompt_builder as prompt_builder

    definition = _definition()
    definition.capabilities.execution_location = ExecutionLocation.LOCAL
    definition.capabilities.filesystem_semantics = FilesystemSemantics.HOST
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    monkeypatch.setattr(prompt_builder, "_probe_remote_backend", lambda _name: None)
    prompt_builder._clear_backend_probe_cache()

    hint = prompt_builder.build_environment_hints()

    assert "Host:" in hint
    assert "User home directory:" in hint
    assert "Current working directory:" in hint


def test_plugin_formats_mocked_remote_probe(monkeypatch):
    import agent.prompt_builder as prompt_builder
    from tools.environments.local import LocalEnvironment

    definition = _definition()
    definition.factory = lambda _request: LocalEnvironment(cwd="/home/coder")
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    probe = MagicMock(return_value="OS: Linux\nUser: coder\nHome: /home/coder")
    monkeypatch.setattr(prompt_builder, "_probe_remote_backend", probe)

    hint = prompt_builder.build_environment_hints()

    probe.assert_called_once_with(definition.name)
    assert "User: coder" in hint
    assert "Home: /home/coder" in hint


@pytest.mark.parametrize(
    "execution_location",
    [ExecutionLocation.REMOTE, ExecutionLocation.UNKNOWN],
)
def test_nonlocal_plugin_cannot_probe_host_through_local_environment(
    monkeypatch, tmp_path, execution_location
):
    import agent.prompt_builder as prompt_builder
    from tools.environments.local import LocalEnvironment

    execute_calls = 0

    class _HostProbeEnvironment(LocalEnvironment):
        def execute(self, *_args, **_kwargs):
            nonlocal execute_calls
            execute_calls += 1
            return {
                "output": "home=/host-secret-sentinel",
                "returncode": 0,
            }

    definition = _definition()
    definition.capabilities.execution_location = execution_location
    definition.factory = lambda _request: _HostProbeEnvironment(cwd=str(tmp_path))
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    prompt_builder._clear_backend_probe_cache()

    hint = prompt_builder.build_environment_hints()

    assert execute_calls == 0
    assert "/host-secret-sentinel" not in hint


def test_host_filesystem_uses_host_path_resolution(monkeypatch):
    from tools.file_tools import _uses_container_paths

    definition = _definition()
    definition.capabilities.execution_location = ExecutionLocation.LOCAL
    definition.capabilities.filesystem_semantics = FilesystemSemantics.HOST
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)

    assert _uses_container_paths() is False


def test_file_paths_follow_isolated_filesystem_semantics(monkeypatch):
    import tools.terminal_tool as terminal_tool
    from tools.environments.local import LocalEnvironment
    from tools.file_tools import (
        _resolve_path_for_task,
        _terminal_env_type_for_task,
        _uses_container_paths,
    )

    definition = _definition()
    definition.default_cwd = "~"
    terminal_backend_registry.register(definition)
    monkeypatch.setenv("TERMINAL_ENV", definition.name)
    monkeypatch.setenv("TERMINAL_CWD", "/opt/host-only/project")
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": LocalEnvironment(cwd="/opt/host-only/project")},
    )

    assert _terminal_env_type_for_task() == definition.name
    assert _uses_container_paths() is True
    assert str(_resolve_path_for_task("evidence.txt")) == "~/evidence.txt"
    assert str(_resolve_path_for_task("~")) == "~"
    assert str(_resolve_path_for_task("~/evidence.txt")) == "~/evidence.txt"
    assert (
        str(_resolve_path_for_task("/workspace/evidence.txt"))
        == "/workspace/evidence.txt"
    )


@pytest.mark.parametrize(
    "backend_name",
    ["docker", "singularity", "modal", "daytona", "vercel_sandbox", "ssh"],
)
def test_builtin_file_tilde_preserves_existing_expansion(
    monkeypatch, backend_name
):
    import tools.file_tools as file_tools

    monkeypatch.setenv("TERMINAL_ENV", backend_name)
    monkeypatch.setattr(
        file_tools,
        "_expand_tilde",
        lambda path: path.replace("~", "/host-home", 1),
    )

    assert str(file_tools._resolve_path_for_task("~/evidence.txt")) == (
        "/host-home/evidence.txt"
    )


def test_threaded_process_handle_is_public_plugin_contract():
    from tools.environments import ThreadedProcessHandle

    handle = ThreadedProcessHandle(lambda: ("ok", 0))
    assert handle.wait(timeout=1) == 0


@pytest.mark.parametrize(
    "backend_name",
    [
        "local",
        "docker",
        "singularity",
        "modal",
        "managed_modal",
        "daytona",
        "ssh",
        "vercel_sandbox",
    ],
)
@pytest.mark.parametrize("source", ["user", "project", "entrypoint"])
def test_plugin_context_rejects_reserved_builtin_backend_names(backend_name, source):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments.registry import BackendAlreadyRegisteredError

    context = PluginContext(
        PluginManifest(name="third-party", source=source), PluginManager()
    )

    definition = _definition()
    definition.name = backend_name
    with pytest.raises(BackendAlreadyRegisteredError, match="reserved"):
        context.register_terminal_backend(definition)



def test_forward_env_helper_preserves_explicit_opt_in_and_filters_implicit_secrets(
    monkeypatch,
):
    from tools.environments import collect_forwarded_env_values
    import tools.env_passthrough as env_passthrough

    monkeypatch.setattr(
        env_passthrough,
        "get_all_passthrough",
        lambda: ["OPENAI_API_KEY", "SAFE_IMPLICIT", "AUXILIARY_TEST_API_KEY"],
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SAFE_IMPLICIT", raising=False)
    monkeypatch.delenv("EXPLICIT_VALUE", raising=False)

    resolved = collect_forwarded_env_values(
        ["OPENAI_API_KEY", "EXPLICIT_VALUE"],
        config_name="test_forward_env",
        dotenv_loader=lambda: {
            "OPENAI_API_KEY": "explicit-provider-secret",
            "SAFE_IMPLICIT": "safe-value",
            "AUXILIARY_TEST_API_KEY": "internal-secret",
            "EXPLICIT_VALUE": "explicit-value",
        },
    )

    assert resolved == {
        "EXPLICIT_VALUE": "explicit-value",
        "OPENAI_API_KEY": "explicit-provider-secret",
        "SAFE_IMPLICIT": "safe-value",
    }


def test_forward_env_helper_normalizes_untrusted_implicit_names(monkeypatch):
    from tools.environments import collect_forwarded_env_values
    import tools.env_passthrough as env_passthrough

    monkeypatch.setattr(
        env_passthrough,
        "get_all_passthrough",
        lambda: [
            " SAFE_IMPLICIT ",
            "SAFE_IMPLICIT",
            "BAD-NAME",
            "OPENAI_API_KEY",
            ["UNHASHABLE"],
            42,
        ],
    )
    monkeypatch.delenv("SAFE_IMPLICIT", raising=False)

    resolved = collect_forwarded_env_values(
        None,
        dotenv_loader=lambda: {
            "SAFE_IMPLICIT": "safe-value",
            "OPENAI_API_KEY": "must-not-forward",
        },
    )

    assert resolved == {"SAFE_IMPLICIT": "safe-value"}


def test_ship_file_to_remote_raises_when_backend_write_fails():
    from tools.code_execution_tool import _ship_file_to_remote

    env = MagicMock()
    env.execute.return_value = {"returncode": 23, "output": "write failed"}

    with pytest.raises(RuntimeError, match="23"):
        _ship_file_to_remote(env, "/tmp/code.py", "print('hello')")


def test_execute_code_resolves_overrides_with_raw_task_id(monkeypatch):
    import threading
    import tools.code_execution_tool as code_execution
    import tools.terminal_tool as terminal_tool

    captured: dict[str, object] = {}
    fake_env = MagicMock(spec=BaseEnvironment)

    def create_environment(**kwargs):
        captured.update(kwargs)
        return fake_env

    resolve_overrides = MagicMock(return_value={"cwd": "/session/workspace"})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_env_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(
        terminal_tool, "_resolve_container_task_id", lambda _task: "collapsed"
    )
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", resolve_overrides)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/global/workspace",
            "timeout": 60,
            "local_persistent": False,
            "host_cwd": None,
        },
    )

    env, env_type = code_execution._get_or_create_env("raw-session")

    assert env is fake_env
    assert env_type == "local"
    resolve_overrides.assert_called_once_with("raw-session")
    assert captured["cwd"] == "/session/workspace"
    assert captured["task_id"] == "collapsed"


def test_manager_resolves_plugin_config_before_calling_factory():
    from tools.environments.manager import EnvironmentManager

    received: dict[str, object] = {}

    def factory(request: BackendFactoryRequest):
        received.update(request.backend_config)
        return MagicMock(spec=BaseEnvironment)

    definition = _definition()
    definition.factory = factory
    definition.config_resolver = lambda raw: {
        **raw,
        "url": "https://coder.example",
        "workspace": "from-environment",
    }
    terminal_backend_registry.register(definition)

    EnvironmentManager().create_environment(
        BackendFactoryRequest(
            backend_name=definition.name,
            backend_config={"workspace": "explicit-override"},
        )
    )

    assert received == {
        "url": "https://coder.example",
        "workspace": "from-environment",
    }


def test_manager_config_resolver_receives_defensive_raw_snapshot():
    from tools.environments.manager import EnvironmentManager

    raw_config = {
        "workspace": "from-yaml",
        "forward_env": ["GITHUB_TOKEN"],
        "nested": {"mode": "yaml"},
    }
    definition = _definition()

    def resolver(raw):
        raw["workspace"] = "from-environment"
        raw["forward_env"].append("SHELL")
        raw["nested"]["mode"] = "resolver"
        return raw

    definition.config_resolver = resolver

    assert EnvironmentManager.resolve_backend_config(definition, raw_config) == {
        "workspace": "from-environment",
        "forward_env": ["GITHUB_TOKEN", "SHELL"],
        "nested": {"mode": "resolver"},
    }
    assert raw_config == {
        "workspace": "from-yaml",
        "forward_env": ["GITHUB_TOKEN"],
        "nested": {"mode": "yaml"},
    }


def test_manager_without_config_resolver_copies_raw_backend_config():
    from tools.environments.manager import EnvironmentManager

    raw_config = {
        "workspace": "from-yaml",
        "forward_env": ["GITHUB_TOKEN"],
        "nested": {"mode": "yaml"},
    }
    resolved = EnvironmentManager.resolve_backend_config(_definition(), raw_config)

    assert resolved == raw_config
    assert resolved is not raw_config
    resolved["forward_env"].append("SHELL")
    resolved["nested"]["mode"] = "resolved"
    assert raw_config["forward_env"] == ["GITHUB_TOKEN"]
    assert raw_config["nested"] == {"mode": "yaml"}


def test_terminal_backend_config_reads_selected_profile_namespace(monkeypatch):
    import hermes_cli.config as config_module
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        config_module,
        "read_raw_config",
        lambda: {
            "terminal": {
                "backends": {
                    "coder": {
                        "base_url": "https://coder.example",
                        "workspace_name": "configured-workspace",
                        "forward_env": ["GITHUB_TOKEN"],
                        "workspace_startup_timeout": 500,
                    }
                }
            }
        },
    )

    assert terminal_tool._backend_config_from_raw_config("coder") == {
        "base_url": "https://coder.example",
        "workspace_name": "configured-workspace",
        "forward_env": ["GITHUB_TOKEN"],
        "workspace_startup_timeout": 500,
    }


def test_terminal_first_use_forwards_profile_backend_config(monkeypatch):
    import tools.terminal_tool as terminal_tool

    task_id = "configured-plugin-terminal-task"
    received: dict[str, object] = {}

    class FakeEnv:
        cwd = "~"
        env = {}

        def execute(self, _command, **_kwargs):
            return {"output": "ok", "returncode": 0}

    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "coder",
            "cwd": "~",
            "host_cwd": None,
            "timeout": 60,
            "lifetime_seconds": 3600,
            "backend_config": {"workspace_name": "terminal-workspace"},
        },
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    def create_environment(**kwargs):
        received.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert received["backend_config"] == {
        "workspace_name": "terminal-workspace"
    }


def test_terminal_preflight_keeps_explicitly_configured_plugin_tools(monkeypatch):
    import tools.terminal_tool as terminal_tool

    definition = _definition()
    definition.availability_check = lambda: False
    definition.config_resolver = lambda raw: {**raw, "api_key": "secret-token"}
    definition.config_availability_check = lambda config: all(
        config.get(key) for key in ("base_url", "api_key", "workspace_name")
    )
    terminal_backend_registry.register(definition)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": definition.name,
            "backend_config": {
                "base_url": "https://coder.example",
                "workspace_name": "configured-workspace",
            },
        },
    )

    assert terminal_tool.check_terminal_requirements() is True


def test_create_environment_forwards_profile_backend_config(monkeypatch):
    import tools.environments.manager as manager_module
    import tools.terminal_tool as terminal_tool

    captured: dict[str, object] = {}
    environment = MagicMock()

    class _Registry:
        @staticmethod
        def get(_name):
            return object()

    class _Manager:
        registry = _Registry()

        @staticmethod
        def create_environment(request):
            captured["request"] = request
            return environment

    monkeypatch.setattr(manager_module, "EnvironmentManager", _Manager)

    result = terminal_tool._create_environment(
        env_type="coder",
        image="",
        cwd="~",
        timeout=180,
        backend_config={
            "base_url": "https://coder.example",
            "workspace_name": "configured-workspace",
        },
    )

    assert result is environment
    request = captured["request"]
    assert isinstance(request, BackendFactoryRequest)
    assert request.backend_config == {
        "base_url": "https://coder.example",
        "workspace_name": "configured-workspace",
    }


def test_manager_accepts_explicit_config_for_env_unavailable_plugin():
    from tools.environments.manager import EnvironmentManager

    received: dict[str, object] = {}

    def factory(request: BackendFactoryRequest):
        received.update(request.backend_config)
        return MagicMock(spec=BaseEnvironment)

    definition = _definition()
    definition.factory = factory
    definition.availability_check = lambda: False
    definition.config_availability_check = lambda config: bool(config)
    definition.config_resolver = lambda raw: {**raw, "api_key": "secret-token"}
    terminal_backend_registry.register(definition)

    EnvironmentManager().create_environment(
        BackendFactoryRequest(
            backend_name=definition.name,
            backend_config={
                "url": "https://coder.example",
                "workspace": "configured-workspace",
            },
        )
    )

    assert received == {
        "api_key": "secret-token",
        "url": "https://coder.example",
        "workspace": "configured-workspace",
    }


def test_manager_preserves_factory_request_cwd_and_host_cwd():
    from tools.environments.manager import EnvironmentManager

    received: dict[str, object] = {}

    def factory(request: BackendFactoryRequest):
        received["cwd"] = request.cwd
        received["host_cwd"] = request.host_cwd
        return MagicMock(spec=BaseEnvironment)

    definition = _definition()
    definition.default_cwd = "~"
    definition.factory = factory
    terminal_backend_registry.register(definition)

    EnvironmentManager().create_environment(
        BackendFactoryRequest(
            backend_name=definition.name,
            cwd="/opt/host-only/project",
            host_cwd="/opt/host-only/project",
        )
    )

    assert received == {
        "cwd": "/opt/host-only/project",
        "host_cwd": "/opt/host-only/project",
    }



def test_manager_accepts_local_plugin_factory_returning_local_environment():
    from tools.environments.local import LocalEnvironment
    from tools.environments.manager import EnvironmentManager

    definition = _definition()
    definition.capabilities.execution_location = ExecutionLocation.LOCAL
    definition.capabilities.filesystem_semantics = FilesystemSemantics.HOST
    definition.factory = lambda _request: LocalEnvironment(cwd="/opt/host-only")
    terminal_backend_registry.register(definition)

    environment = EnvironmentManager().create_environment(
        BackendFactoryRequest(backend_name=definition.name)
    )

    assert isinstance(environment, LocalEnvironment)


def test_manager_rejects_remote_plugin_factory_returning_local_environment():
    from tools.environments.local import LocalEnvironment
    from tools.environments.manager import EnvironmentManager

    definition = _definition()
    definition.factory = lambda _request: LocalEnvironment(cwd="/opt/host-only")
    terminal_backend_registry.register(definition)

    with pytest.raises(TypeError, match="declares remote execution"):
        EnvironmentManager().create_environment(
            BackendFactoryRequest(backend_name=definition.name)
        )


def test_manager_rejects_non_mapping_plugin_config():
    from tools.environments.manager import EnvironmentManager

    definition = _definition()
    definition.config_resolver = lambda raw: ["not", "a", "mapping"]
    terminal_backend_registry.register(definition)

    with pytest.raises(TypeError, match="config_resolver must return a mapping"):
        EnvironmentManager().create_environment(
            BackendFactoryRequest(backend_name=definition.name)
        )
