from __future__ import annotations

import pytest


def test_environment_manager_bootstraps_builtin_local_backend():
    from tools.environments import ExecutionLocation, FilesystemSemantics
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import TerminalBackendRegistry

    registry = TerminalBackendRegistry()
    manager = EnvironmentManager(registry=registry)

    definition = registry.require("local")
    assert manager.registry is registry
    assert definition.name == "local"
    assert definition.label == "Local"
    assert definition.source == "builtin"
    assert definition.plugin_name == ""
    assert definition.capabilities.execution_location is ExecutionLocation.LOCAL
    assert definition.capabilities.filesystem_semantics is FilesystemSemantics.HOST
    assert definition.capabilities.accepts_host_cwd is True
    assert definition.capabilities.requires_sandbox_cwd is False
    assert definition.capabilities.supports_image is False
    assert definition.capabilities.supports_resource_limits is False
    assert definition.capabilities.supports_pty is True
    assert definition.capabilities.supports_background_processes is True
    assert definition.capabilities.supports_file_transfer is False
    assert definition.capabilities.supports_persistence is True


def test_environment_manager_creates_and_executes_builtin_local_backend(tmp_path):
    from tools.environments import BackendFactoryRequest
    from tools.environments.local import LocalEnvironment
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import TerminalBackendRegistry

    manager = EnvironmentManager(registry=TerminalBackendRegistry())
    request = BackendFactoryRequest(
        backend_name="local",
        task_id="local-task",
        cwd=str(tmp_path),
        timeout=17,
        image="ignored-for-local",
        host_cwd=str(tmp_path),
        terminal_config={"local_config": {"inherit_env": False}},
    )

    environment = manager.create_environment(request)
    try:
        assert isinstance(environment, LocalEnvironment)
        assert environment.cwd == str(tmp_path)
        assert environment.timeout == 17

        result = environment.execute("printf registry-local")
        assert result["returncode"] == 0
        assert "registry-local" in result["output"]
    finally:
        environment.cleanup()



def test_environment_manager_rejects_non_environment_factory_result():
    from tools.environments import BackendDefinition, BackendFactoryRequest
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import TerminalBackendRegistry

    registry = TerminalBackendRegistry()
    manager = EnvironmentManager(registry=registry)
    registry.register(
        BackendDefinition(name="invalid", factory=lambda request: object())
    )

    with pytest.raises(TypeError, match="BaseEnvironment"):
        manager.create_environment(BackendFactoryRequest(backend_name="invalid"))


def test_environment_manager_rejects_unavailable_backend_before_factory_call():
    from tools.environments import BackendDefinition, BackendFactoryRequest
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import (
        BackendUnavailableError,
        TerminalBackendRegistry,
    )

    factory_called = False

    def factory(request):
        nonlocal factory_called
        factory_called = True
        return object()

    registry = TerminalBackendRegistry()
    manager = EnvironmentManager(registry=registry)
    registry.register(
        BackendDefinition(
            name="unavailable",
            factory=factory,
            availability_check=lambda: False,
        )
    )

    with pytest.raises(BackendUnavailableError, match="unavailable"):
        manager.create_environment(BackendFactoryRequest(backend_name="unavailable"))

    assert factory_called is False
