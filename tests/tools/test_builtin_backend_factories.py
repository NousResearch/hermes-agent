from __future__ import annotations

import pytest

from tools.environments import BackendFactoryRequest
from tools.environments.base import BaseEnvironment
from tools.environments.manager import EnvironmentManager
from tools.environments.registry import TerminalBackendRegistry


class _CapturedEnvironment(BaseEnvironment):
    def __init__(self, **kwargs):
        super().__init__(cwd=kwargs["cwd"], timeout=kwargs["timeout"])
        self.kwargs = kwargs

    def _run_bash(self, *args, **kwargs):
        raise AssertionError("factory wiring test must not execute commands")

    def cleanup(self):
        return None



def _request(name: str, **overrides) -> BackendFactoryRequest:
    values = {
        "backend_name": name,
        "image": "example/image:latest",
        "cwd": "/sandbox/work",
        "timeout": 37,
        "task_id": "task-123",
        "host_cwd": "/host/work",
        "terminal_config": {},
    }
    values.update(overrides)
    return BackendFactoryRequest(**values)


def _create_via_manager(request: BackendFactoryRequest) -> BaseEnvironment:
    manager = EnvironmentManager(registry=TerminalBackendRegistry())
    return manager.create_environment(request)


def test_terminal_factory_unknown_backend_preserves_legacy_error_contract():
    from tools.terminal_tool import _create_environment

    with pytest.raises(
        ValueError,
        match=(
            "Unknown environment type: bogus.*'local'.*'docker'.*"
            "'singularity'.*'modal'.*'daytona'.*'vercel_sandbox'.*'ssh'"
        ),
    ):
        _create_environment(
            "bogus",
            image="",
            cwd="~",
            timeout=180,
        )


def test_docker_factory_preserves_every_legacy_constructor_option(monkeypatch):
    import tools.environments.docker as docker_backend
    import tools.terminal_tool as terminal_tool

    reaped = []
    monkeypatch.setattr(docker_backend, "DockerEnvironment", _CapturedEnvironment)
    monkeypatch.setattr(
        terminal_tool, "_maybe_reap_docker_orphans", lambda config: reaped.append(config)
    )
    container_config = {
        "container_cpu": 2.5,
        "container_memory": 6144,
        "container_disk": 32768,
        "container_persistent": False,
        "docker_volumes": ["/a:/b"],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["TOKEN"],
        "docker_env": {"MODE": "test"},
        "docker_run_as_host_user": True,
        "docker_network": False,
        "docker_extra_args": ["--cap-drop=ALL"],
        "docker_persist_across_processes": False,
        "docker_shm_size": "2g",
    }

    environment = _create_via_manager(
        _request(
            "docker",
            terminal_config={"container_config": container_config},
        )
    )

    assert reaped == [container_config]
    assert environment.kwargs == {
        "image": "example/image:latest",
        "cwd": "/sandbox/work",
        "timeout": 37,
        "cpu": 2.5,
        "memory": 6144,
        "disk": 32768,
        "persistent_filesystem": False,
        "task_id": "task-123",
        "volumes": ["/a:/b"],
        "host_cwd": "/host/work",
        "auto_mount_cwd": True,
        "forward_env": ["TOKEN"],
        "env": {"MODE": "test"},
        "run_as_host_user": True,
        "network": False,
        "extra_args": ["--cap-drop=ALL"],
        "persist_across_processes": False,
        "shm_size": "2g",
    }



@pytest.mark.parametrize(
    ("name", "module_name", "class_name", "expected"),
    [
        (
            "singularity",
            "tools.environments.singularity",
            "SingularityEnvironment",
            {"cpu": 2.5, "memory": 6144, "disk": 32768},
        ),
        (
            "daytona",
            "tools.environments.daytona",
            "DaytonaEnvironment",
            {"cpu": 2, "memory": 6144, "disk": 32768},
        ),
        (
            "vercel_sandbox",
            "tools.environments.vercel_sandbox",
            "VercelSandboxEnvironment",
            {
                "runtime": "python3.13",
                "cpu": 2.5,
                "memory": 6144,
                "disk": 32768,
            },
        ),
    ],
)
def test_container_factory_preserves_legacy_resource_mapping(
    monkeypatch, name, module_name, class_name, expected
):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, class_name, _CapturedEnvironment)
    container_config = {
        "container_cpu": 2.5,
        "container_memory": 6144,
        "container_disk": 32768,
        "container_persistent": False,
        "vercel_runtime": "python3.13",
    }

    environment = _create_via_manager(
        _request(name, terminal_config={"container_config": container_config})
    )

    common = {
        "cwd": "/sandbox/work",
        "timeout": 37,
        "persistent_filesystem": False,
        "task_id": "task-123",
    }
    if name != "vercel_sandbox":
        common["image"] = "example/image:latest"
    assert environment.kwargs == common | expected


def test_ssh_factory_preserves_legacy_connection_mapping(monkeypatch):
    import tools.environments.ssh as ssh_backend

    monkeypatch.setattr(ssh_backend, "SSHEnvironment", _CapturedEnvironment)
    environment = _create_via_manager(
        _request(
            "ssh",
            terminal_config={
                "ssh_config": {
                    "host": "example.test",
                    "user": "hermes",
                    "port": 2202,
                    "key": "/keys/id",
                    "persistent": True,
                }
            },
        )
    )

    assert environment.kwargs == {
        "host": "example.test",
        "user": "hermes",
        "port": 2202,
        "key_path": "/keys/id",
        "cwd": "/sandbox/work",
        "timeout": 37,
    }


def test_ssh_factory_rejects_missing_required_connection_fields():
    with pytest.raises(ValueError, match="ssh_host and ssh_user"):
        _create_via_manager(
            _request("ssh", terminal_config={"ssh_config": {"host": "example.test"}})
        )


@pytest.mark.parametrize(
    ("selected", "module_name", "class_name"),
    [
        ("direct", "tools.environments.modal", "ModalEnvironment"),
        ("managed", "tools.environments.managed_modal", "ManagedModalEnvironment"),
    ],
)
def test_modal_factory_keeps_direct_and_managed_as_one_public_backend(
    monkeypatch, selected, module_name, class_name
):
    import importlib
    import tools.managed_tool_gateway as gateway
    import tools.tool_backend_helpers as helpers
    from tools.environments import builtin_backends

    target_module = importlib.import_module(module_name)
    monkeypatch.setattr(target_module, class_name, _CapturedEnvironment)
    monkeypatch.setattr(
        helpers,
        "resolve_modal_backend_state",
        lambda *args, **kwargs: {
            "selected_backend": selected,
            "managed_mode_blocked": False,
            "mode": "auto",
        },
    )
    monkeypatch.setattr(helpers, "has_direct_modal_credentials", lambda: True)
    monkeypatch.setattr(gateway, "is_managed_tool_gateway_ready", lambda _name: True)
    monkeypatch.setattr(
        builtin_backends,
        "_modal_sandbox_kwargs",
        lambda _config: {"cpu": 2.5},
    )

    environment = _create_via_manager(
        _request(
            "modal",
            terminal_config={
                "container_config": {
                    "modal_mode": "auto",
                    "container_persistent": False,
                }
            },
        )
    )

    assert environment.kwargs == {
        "image": "example/image:latest",
        "cwd": "/sandbox/work",
        "timeout": 37,
        "modal_sandbox_kwargs": {"cpu": 2.5},
        "persistent_filesystem": False,
        "task_id": "task-123",
    }


def test_modal_manager_uses_request_auto_mode_for_first_creator(monkeypatch):
    import tools.environments.modal as modal_backend
    import tools.managed_tool_gateway as gateway
    import tools.tool_backend_helpers as helpers
    from tools.environments import builtin_backends

    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setattr(modal_backend, "ModalEnvironment", _CapturedEnvironment)
    monkeypatch.setattr(helpers, "has_direct_modal_credentials", lambda: True)
    monkeypatch.setattr(gateway, "is_managed_tool_gateway_ready", lambda _name: False)
    monkeypatch.setattr(builtin_backends, "_modal_sandbox_kwargs", lambda _config: {})

    environment = _create_via_manager(_request("modal", terminal_config={}))

    assert isinstance(environment, _CapturedEnvironment)


def test_modal_manager_preserves_backend_specific_configuration_error(monkeypatch):
    import tools.managed_tool_gateway as gateway
    import tools.tool_backend_helpers as helpers

    monkeypatch.setattr(helpers, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(helpers, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(gateway, "is_managed_tool_gateway_ready", lambda _name: False)

    with pytest.raises(
        ValueError,
        match="no direct Modal credentials/config was found",
    ):
        _create_via_manager(_request("modal", terminal_config={}))
