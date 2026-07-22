from __future__ import annotations

import pytest

from tools.environments import BackendFactoryRequest


@pytest.fixture(autouse=True)
def _reset_facade():
    from tools.environments.facade import reset_environment_facade

    reset_environment_facade()
    yield
    reset_environment_facade()


def test_environment_facade_routes_to_legacy_runtime_by_default(monkeypatch):
    from tools.environments.facade import (
        LegacyEnvironmentFacade,
        get_environment_facade,
    )

    monkeypatch.delenv("EXP_BACKEND", raising=False)
    sentinel = object()
    requests = []

    def legacy_factory(request):
        requests.append(request)
        return sentinel

    request = BackendFactoryRequest(backend_name="local", cwd="/workspace")
    facade = get_environment_facade(legacy_factory=legacy_factory)

    assert isinstance(facade, LegacyEnvironmentFacade)
    assert facade.create_environment(request) is sentinel
    assert requests == [request]


def test_environment_facade_routes_to_manager_when_exp_backend_is_one(monkeypatch):
    from tools.environments.facade import (
        ExperimentalEnvironmentFacade,
        get_environment_facade,
    )

    monkeypatch.setenv("EXP_BACKEND", "1")
    sentinel = object()
    requests = []

    class FakeManager:
        def create_environment(self, request):
            requests.append(request)
            return sentinel

    def legacy_factory(request):
        raise AssertionError("legacy runtime must not be called")

    request = BackendFactoryRequest(backend_name="coder", task_id="task-1")
    facade = get_environment_facade(
        legacy_factory=legacy_factory,
        manager_factory=FakeManager,
    )

    assert isinstance(facade, ExperimentalEnvironmentFacade)
    assert facade.create_environment(request) is sentinel
    assert requests == [request]


def test_terminal_factory_enters_facade_and_preserves_legacy_arguments(monkeypatch):
    from tools.environments.facade import reset_environment_facade
    from tools import terminal_tool

    monkeypatch.delenv("EXP_BACKEND", raising=False)
    reset_environment_facade()
    sentinel = object()
    captured = {}

    def legacy_factory(
        env_type,
        image,
        cwd,
        timeout,
        ssh_config=None,
        container_config=None,
        local_config=None,
        task_id="default",
        host_cwd=None,
    ):
        captured.update(
            env_type=env_type,
            image=image,
            cwd=cwd,
            timeout=timeout,
            ssh_config=ssh_config,
            container_config=container_config,
            local_config=local_config,
            task_id=task_id,
            host_cwd=host_cwd,
        )
        return sentinel

    monkeypatch.setattr(terminal_tool, "_create_environment_legacy", legacy_factory)

    result = terminal_tool._create_environment(
        "local",
        "python:3.13",
        "/workspace",
        30,
        ssh_config={"host": "example"},
        container_config={"container_cpu": 2},
        local_config={"inherit_env": False},
        task_id="task-1",
        host_cwd="/host/project",
    )

    assert result is sentinel
    assert captured == {
        "env_type": "local",
        "image": "python:3.13",
        "cwd": "/workspace",
        "timeout": 30,
        "ssh_config": {"host": "example"},
        "container_config": {"container_cpu": 2},
        "local_config": {"inherit_env": False},
        "task_id": "task-1",
        "host_cwd": "/host/project",
    }


def test_environment_facade_freezes_runtime_selection_for_the_process(monkeypatch):
    from tools.environments.facade import get_environment_facade

    monkeypatch.delenv("EXP_BACKEND", raising=False)
    legacy = get_environment_facade(legacy_factory=lambda request: object())

    monkeypatch.setenv("EXP_BACKEND", "1")
    selected_again = get_environment_facade(
        legacy_factory=lambda request: object(),
        manager_factory=lambda: pytest.fail("runtime selection changed"),
    )

    assert selected_again is legacy


def test_exp_backend_zero_uses_legacy_without_warning(monkeypatch, caplog):
    from tools.environments.facade import (
        LegacyEnvironmentFacade,
        get_environment_facade,
    )

    monkeypatch.setenv("EXP_BACKEND", "0")

    with caplog.at_level("WARNING"):
        facade = get_environment_facade(legacy_factory=lambda request: object())

    assert isinstance(facade, LegacyEnvironmentFacade)
    assert "EXP_BACKEND" not in caplog.text


@pytest.mark.parametrize("invalid_value", ["", " ", "true"])
def test_invalid_exp_backend_value_warns_and_uses_legacy(
    monkeypatch, caplog, invalid_value
):
    from tools.environments.facade import (
        LegacyEnvironmentFacade,
        get_environment_facade,
    )

    monkeypatch.setenv("EXP_BACKEND", invalid_value)

    with caplog.at_level("WARNING"):
        facade = get_environment_facade(legacy_factory=lambda request: object())

    assert isinstance(facade, LegacyEnvironmentFacade)
    assert "EXP_BACKEND" in caplog.text
    assert "legacy" in caplog.text


def test_terminal_factory_uses_unimplemented_manager_when_exp_backend_is_one(
    monkeypatch,
):
    from tools.environments.facade import reset_environment_facade
    from tools import terminal_tool

    monkeypatch.setenv("EXP_BACKEND", "1")
    reset_environment_facade()
    monkeypatch.setattr(
        terminal_tool,
        "_create_environment_legacy",
        lambda *args, **kwargs: pytest.fail("legacy runtime was called"),
    )

    with pytest.raises(NotImplementedError, match="experimental backend runtime"):
        terminal_tool._create_environment("coder", "", "/workspace", 30)
