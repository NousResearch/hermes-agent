"""Regression coverage for the shared sandbox environment builders."""

from types import SimpleNamespace

import pytest

from tools import terminal_tool_backends as backends


class _FakeEnvironment:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("env_type", "module_name", "class_name", "image", "container_config"),
    [
        ("singularity", None, None, "test.sif", {}),
        ("daytona", "tools.environments.daytona", "DaytonaEnvironment", "test:latest", {"container_cpu": 2}),
        (
            "vercel_sandbox",
            "tools.environments.vercel_sandbox",
            "VercelSandboxEnvironment",
            "ignored",
            {"vercel_runtime": "node24"},
        ),
    ],
)
def test_create_environment_dispatches_sandbox_backends_without_duplicate_env_type(
    monkeypatch, env_type, module_name, class_name, image, container_config
):
    """The dispatch path must pass ``env_type`` exactly once to sandbox builders."""
    if env_type == "singularity":
        monkeypatch.setattr(backends, "_SingularityEnvironment", _FakeEnvironment)
    else:
        real_import_module = backends.importlib.import_module

        def fake_import_module(name):
            if name == module_name:
                return SimpleNamespace(**{class_name: _FakeEnvironment})
            return real_import_module(name)

        monkeypatch.setattr(backends.importlib, "import_module", fake_import_module)

    env = backends._create_environment(
        env_type=env_type,
        image=image,
        cwd="/workspace",
        timeout=60,
        container_config=container_config,
        task_id="test-task",
    )

    assert isinstance(env, _FakeEnvironment)
    assert env.kwargs["cwd"] == "/workspace"
    assert env.kwargs["timeout"] == 60
    assert env.kwargs["task_id"] == "test-task"

    if env_type == "vercel_sandbox":
        assert "image" not in env.kwargs
        assert env.kwargs["runtime"] == "node24"
    else:
        assert env.kwargs["image"] == image

    if env_type == "daytona":
        assert env.kwargs["cpu"] == 2
