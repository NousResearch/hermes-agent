from __future__ import annotations

from pathlib import Path

import pytest


def _factory(request):
    return object()


def test_backend_definition_is_public_and_mutable():
    from tools.environments import (
        BackendCapabilities,
        BackendDefinition,
        ExecutionLocation,
        FilesystemSemantics,
    )

    capabilities = BackendCapabilities(
        execution_location=ExecutionLocation.REMOTE,
        filesystem_semantics=FilesystemSemantics.ISOLATED,
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

    definition.name = "other"
    capabilities.supports_pty = False

    assert definition.name == "other"
    assert definition.capabilities.supports_pty is False


def test_backend_capabilities_separate_declaration_effective_and_runtime_state():
    from dataclasses import fields

    from tools.environments import BackendCapabilities
    from tools.environments.definitions import (
        EffectiveBackendCapabilities,
        EnvironmentRuntimeState,
        FilesystemSemantics,
        HostAccess,
    )

    assert "host_access" not in {field.name for field in fields(BackendCapabilities)}

    declared = BackendCapabilities(supports_pty=True)
    effective = EffectiveBackendCapabilities(
        supports_pty=declared.supports_pty,
        filesystem_semantics=FilesystemSemantics.ISOLATED,
        host_access=HostAccess.POSSIBLE,
    )
    runtime = EnvironmentRuntimeState(
        backend_name="docker",
        task_id="task-1",
        filesystem_semantics=FilesystemSemantics.ISOLATED,
        host_access=HostAccess.NONE,
        isolation_verified_by_host=True,
    )

    unverified_runtime = EnvironmentRuntimeState(
        backend_name="plugin-backend",
        task_id="task-2",
        host_access=HostAccess.NONE,
    )

    assert effective.host_access is HostAccess.POSSIBLE
    assert runtime.host_access is HostAccess.NONE
    assert runtime.has_verified_no_host_access is True
    assert unverified_runtime.has_verified_no_host_access is False


def test_host_owned_capability_state_is_not_reexported_to_plugins():
    import tools.environments as public_contract

    host_owned_names = {
        "HostAccess",
        "EffectiveBackendCapabilities",
        "EnvironmentRuntimeState",
    }

    assert host_owned_names.isdisjoint(public_contract.__all__)
    assert all(not hasattr(public_contract, name) for name in host_owned_names)


@pytest.mark.parametrize(
    ("host_access", "verified", "expected"),
    [
        ("unknown", False, False),
        ("unknown", True, False),
        ("possible", True, False),
        ("direct", True, False),
        ("none", False, False),
        ("none", True, True),
    ],
)
def test_runtime_host_access_state_fails_closed(host_access, verified, expected):
    from tools.environments.definitions import EnvironmentRuntimeState, HostAccess

    state = EnvironmentRuntimeState(
        backend_name="backend",
        task_id="task-1",
        host_access=HostAccess(host_access),
        isolation_verified_by_host=verified,
    )

    assert state.has_verified_no_host_access is expected


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

    request.cwd = "/other"
    assert request.cwd == "/other"


def test_backend_contract_mappings_copy_top_level_and_remain_mutable():
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

    request.terminal_config["container_cpu"] = 8
    definition.config_schema["other"] = {"type": "number"}
    definition.diagnostic_metadata["other"] = False

    assert request.terminal_config["container_cpu"] == 8
    assert definition.config_schema["other"] == {"type": "number"}
    assert definition.diagnostic_metadata["other"] is False


def test_backend_contract_nested_mappings_remain_mutable():
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

    request.terminal_config["ssh"]["options"].append("ForwardAgent=yes")
    request.terminal_config["ssh"]["ports"]["http"] = 9090
    definition.config_schema["workspace"]["examples"].append("/tmp")

    assert request.terminal_config["ssh"]["options"] == [
        "Compression=yes",
        "ForwardAgent=yes",
    ]
    assert request.terminal_config["ssh"]["ports"] == {"http": 9090}
    assert definition.config_schema["workspace"]["examples"] == [
        "/workspace",
        "/tmp",
    ]


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


def test_environment_manager_leaves_unmigrated_lifecycle_surface_unimplemented():
    from typing import get_type_hints

    from tools.environments import BackendFactoryRequest
    from tools.environments.definitions import (
        EffectiveBackendCapabilities,
        EnvironmentRuntimeState,
    )
    from tools.environments.manager import EnvironmentManager
    from tools.environments.registry import (
        BackendNotFoundError,
        TerminalBackendRegistry,
        terminal_backend_registry,
    )

    registry = TerminalBackendRegistry()
    manager = EnvironmentManager(registry=registry)
    request = BackendFactoryRequest(backend_name="coder")

    assert manager.registry is registry
    assert EnvironmentManager().registry is terminal_backend_registry
    assert (
        get_type_hints(EnvironmentManager.get_effective_capabilities)["return"]
        is EffectiveBackendCapabilities
    )
    assert get_type_hints(EnvironmentManager.get_runtime_state)["return"] == (
        EnvironmentRuntimeState | None
    )

    with pytest.raises(BackendNotFoundError, match="coder"):
        manager.resolve_backend("coder")

    operations = [
        lambda: manager.get_or_create_environment(request),
        lambda: manager.get_active_environment("task-1"),
        lambda: manager.get_effective_backend_name("task-1"),
        lambda: manager.get_effective_capabilities("task-1"),
        lambda: manager.get_runtime_state("task-1"),
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
