from __future__ import annotations

import pytest


def _factory(request):
    return object()


def test_backend_definition_validates_public_registration_contract():
    from tools.environments import BackendDefinition

    assert BackendDefinition(name="coder", factory=_factory).label == "coder"

    with pytest.raises(ValueError, match="backend name"):
        BackendDefinition(name="Coder Backend", factory=_factory)

    with pytest.raises(TypeError, match="factory"):
        BackendDefinition(name="coder", factory=None)

    with pytest.raises(TypeError, match="availability_check"):
        BackendDefinition(name="coder", factory=_factory, availability_check=None)


@pytest.mark.parametrize("field", ["label", "description", "install_hint"])
def test_backend_definition_rejects_non_string_picker_metadata(field):
    from tools.environments import BackendDefinition

    with pytest.raises(TypeError, match=rf"{field} must be a string"):
        BackendDefinition(
            name="invalid",
            factory=_factory,
            **{field: object()},
        )


@pytest.mark.parametrize(
    ("config_schema", "error_type", "match"),
    [
        ({"token": "secret"}, TypeError, "must be a mapping"),
        (
            {"token": {"type": "secret", "config_key": 7}},
            TypeError,
            "config_key must be a non-empty string",
        ),
        (
            {"__proto__.polluted": {"type": "string"}},
            ValueError,
            "unsafe path segment",
        ),
        (
            {
                "token": {
                    "type": "secret",
                    "config_key": "terminal.backends.invalid.constructor.value",
                }
            },
            ValueError,
            "unsafe path segment",
        ),
        (
            {"token": {"type": "unsupported"}},
            ValueError,
            "unsupported type",
        ),
        (
            {"mode": {"type": "select", "options": "not-a-list"}},
            TypeError,
            "options must be a list of strings",
        ),
        (
            {"mode": {"type": "select", "options": ["safe", 7]}},
            TypeError,
            "options must be a list of strings",
        ),
        (
            {"token": {"type": "string", "metadata": object()}},
            TypeError,
            "JSON-serializable",
        ),
    ],
)
def test_backend_definition_rejects_invalid_config_schema_entry(
    config_schema, error_type, match
):
    from tools.environments import BackendDefinition

    with pytest.raises(error_type, match=match):
        BackendDefinition(
            name="invalid",
            factory=_factory,
            config_schema=config_schema,
        )


def test_backend_factory_request_carries_runtime_configuration():
    from tools.environments import BackendFactoryRequest

    request = BackendFactoryRequest(
        backend_name="coder",
        task_id="task-1",
        cwd="/workspace",
        timeout=30,
        image="python:3.13",
        host_cwd="/host/project",
        terminal_config={"container_cpu": 2},
        backend_config={"workspace": "example"},
    )

    assert request.backend_name == "coder"
    assert request.task_id == "task-1"
    assert request.cwd == "/workspace"
    assert request.timeout == 30
    assert request.image == "python:3.13"
    assert request.host_cwd == "/host/project"
    assert request.terminal_config == {"container_cpu": 2}
    assert request.backend_config == {"workspace": "example"}

    request.cwd = "/other"
    assert request.cwd == "/other"


def test_backend_contract_mappings_copy_top_level_and_remain_mutable():
    from tools.environments import BackendDefinition, BackendFactoryRequest

    terminal_config = {"container_cpu": 2}
    backend_config = {"workspace": "example"}
    config_schema = {"workspace": {"type": "string"}}
    diagnostic_metadata = {"docs_url": "https://example.invalid"}

    request = BackendFactoryRequest(
        backend_name="coder",
        terminal_config=terminal_config,
        backend_config=backend_config,
    )
    definition = BackendDefinition(
        name="coder",
        factory=_factory,
        config_schema=config_schema,
        diagnostic_metadata=diagnostic_metadata,
    )

    terminal_config["container_cpu"] = 4
    backend_config["workspace"] = "changed"
    config_schema["other"] = {"type": "boolean"}
    diagnostic_metadata["other"] = True

    assert request.terminal_config == {"container_cpu": 2}
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
