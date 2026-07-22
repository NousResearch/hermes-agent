from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def _factory(request):
    return object()


def test_backend_definition_is_public_and_immutable():
    from tools.environments import (
        BackendCapabilities,
        BackendDefinition,
        ExecutionLocation,
        FilesystemSemantics,
        HostAccess,
    )

    capabilities = BackendCapabilities(
        execution_location=ExecutionLocation.REMOTE,
        filesystem_semantics=FilesystemSemantics.ISOLATED,
        host_access=HostAccess.POSSIBLE,
        accepts_host_cwd=False,
        supports_image=True,
        supports_resource_limits=True,
        supports_pty=True,
    )
    definition = BackendDefinition(
        name="coder",
        label="Coder",
        factory=_factory,
        capabilities=capabilities,
    )

    assert definition.name == "coder"
    assert definition.label == "Coder"
    assert definition.capabilities is capabilities
    assert definition.is_available() is True

    with pytest.raises(FrozenInstanceError):
        definition.name = "other"


def test_backend_definition_validates_public_registration_contract():
    from tools.environments import BackendDefinition

    assert BackendDefinition(name="coder", factory=_factory).label == "coder"

    with pytest.raises(ValueError, match="backend name"):
        BackendDefinition(name="Coder Backend", factory=_factory)

    with pytest.raises(TypeError, match="factory"):
        BackendDefinition(name="coder", factory=None)

    with pytest.raises(TypeError, match="availability_check"):
        BackendDefinition(name="coder", factory=_factory, availability_check=None)


def test_backend_factory_request_carries_host_and_backend_configuration():
    from tools.environments import BackendFactoryRequest

    request = BackendFactoryRequest(
        backend_name="coder",
        task_id="task-1",
        cwd="/workspace",
        timeout=30,
        image="python:3.13",
        host_cwd="/host/project",
        profile_name="work",
        hermes_home=Path("/home/example/.hermes"),
        terminal_config={"container_cpu": 2},
        task_overrides={"timeout": 30},
        backend_config={"workspace": "example"},
    )

    assert request.backend_name == "coder"
    assert request.task_id == "task-1"
    assert request.cwd == "/workspace"
    assert request.timeout == 30
    assert request.image == "python:3.13"
    assert request.host_cwd == "/host/project"
    assert request.profile_name == "work"
    assert request.hermes_home == Path("/home/example/.hermes")
    assert request.terminal_config == {"container_cpu": 2}
    assert request.task_overrides == {"timeout": 30}
    assert request.backend_config == {"workspace": "example"}

    with pytest.raises(FrozenInstanceError):
        request.cwd = "/other"


def test_backend_contract_mappings_are_defensive_and_read_only():
    from tools.environments import BackendDefinition, BackendFactoryRequest

    terminal_config = {"container_cpu": 2}
    task_overrides = {"timeout": 30}
    backend_config = {"workspace": "example"}
    config_schema = {"workspace": {"type": "string"}}
    diagnostic_metadata = {"docs_url": "https://example.invalid"}

    request = BackendFactoryRequest(
        backend_name="coder",
        terminal_config=terminal_config,
        task_overrides=task_overrides,
        backend_config=backend_config,
    )
    definition = BackendDefinition(
        name="coder",
        factory=_factory,
        config_schema=config_schema,
        diagnostic_metadata=diagnostic_metadata,
    )

    terminal_config["container_cpu"] = 4
    task_overrides["timeout"] = 60
    backend_config["workspace"] = "changed"
    config_schema["other"] = {"type": "boolean"}
    diagnostic_metadata["other"] = True

    assert request.terminal_config == {"container_cpu": 2}
    assert request.task_overrides == {"timeout": 30}
    assert request.backend_config == {"workspace": "example"}
    assert definition.config_schema == {"workspace": {"type": "string"}}
    assert definition.diagnostic_metadata == {"docs_url": "https://example.invalid"}

    with pytest.raises(TypeError):
        request.terminal_config["container_cpu"] = 8
    with pytest.raises(TypeError):
        definition.config_schema["other"] = {"type": "number"}
    with pytest.raises(TypeError):
        definition.diagnostic_metadata["other"] = False


def test_backend_contract_mappings_are_recursively_defensive_and_read_only():
    from tools.environments import BackendDefinition, BackendFactoryRequest

    terminal_config = {"ssh": {"options": ["Compression=yes"], "ports": {"http": 8080}}}
    config_schema = {"workspace": {"type": "string", "examples": ["/workspace"]}}

    request = BackendFactoryRequest(
        backend_name="coder",
        terminal_config=terminal_config,
    )
    definition = BackendDefinition(
        name="coder",
        factory=_factory,
        config_schema=config_schema,
    )

    terminal_config["ssh"]["options"].append("ForwardAgent=yes")
    terminal_config["ssh"]["ports"]["http"] = 9090
    config_schema["workspace"]["examples"].append("/tmp")

    assert request.terminal_config["ssh"]["options"] == ("Compression=yes",)
    assert request.terminal_config["ssh"]["ports"] == {"http": 8080}
    assert definition.config_schema["workspace"]["examples"] == ("/workspace",)

    with pytest.raises(TypeError):
        request.terminal_config["ssh"]["ports"]["http"] = 9090
    with pytest.raises(AttributeError):
        definition.config_schema["workspace"]["examples"].append("/tmp")


def test_terminal_backend_registry_rejects_duplicate_names():
    from tools.environments import BackendDefinition
    from tools.environments.registry import (
        BackendAlreadyRegisteredError,
        TerminalBackendRegistry,
    )

    registry = TerminalBackendRegistry()
    definition = BackendDefinition(name="coder", factory=_factory)

    registry.register(definition)

    assert registry.require("coder") is definition
    assert registry.list_definitions() == (definition,)

    with pytest.raises(BackendAlreadyRegisteredError, match="coder"):
        registry.register(BackendDefinition(name="coder", factory=_factory))

    registry.reset()
    assert registry.list_definitions() == ()

    with pytest.raises(TypeError, match="BackendDefinition"):
        registry.register(object())


def test_environment_manager_declares_lifecycle_surface_as_unimplemented():
    from tools.environments import BackendFactoryRequest
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import (
        TerminalBackendRegistry,
        terminal_backend_registry,
    )

    registry = TerminalBackendRegistry()
    manager = EnvironmentManager(registry=registry)
    request = BackendFactoryRequest(backend_name="coder")

    assert manager.registry is registry
    assert EnvironmentManager().registry is terminal_backend_registry

    operations = [
        lambda: manager.resolve_backend("coder"),
        lambda: manager.create_environment(request),
        lambda: manager.get_or_create_environment(request),
        lambda: manager.get_active_environment("task-1"),
        lambda: manager.get_effective_backend_name("task-1"),
        lambda: manager.get_capabilities("task-1"),
        lambda: manager.register_task_overrides("task-1", {"backend": "coder"}),
        lambda: manager.resolve_task_overrides("task-1"),
        lambda: manager.clear_task_overrides("task-1"),
        lambda: manager.mark_activity("task-1"),
        lambda: manager.cleanup_environment("task-1"),
        manager.cleanup_all,
        manager.snapshot,
    ]

    for operation in operations:
        with pytest.raises(NotImplementedError, match="experimental backend runtime"):
            operation()
