"""Regression coverage for sandbox backend builder dispatch (#112715)."""

import pytest

from tools import terminal_tool_backends as backends


@pytest.mark.parametrize(
    ("env_type", "expects_image"),
    [
        ("singularity", True),
        ("daytona", True),
        ("vercel_sandbox", False),
    ],
)
def test_create_environment_dispatches_each_sandbox_builder_once(
    monkeypatch, env_type, expects_image
):
    """The bound backend name must not collide with the generic dispatcher kwarg."""
    received = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setitem(
        backends._SANDBOX_ROWS,
        env_type,
        (lambda: FakeEnvironment, expects_image, lambda _cc, _kwargs: {}),
    )

    environment = backends._create_environment(
        env_type=env_type,
        image="test-image",
        cwd="/workspace",
        timeout=42,
        container_config={},
        task_id="test-task",
    )

    assert isinstance(environment, FakeEnvironment)
    assert received["cwd"] == "/workspace"
    assert received["timeout"] == 42
    assert received["task_id"] == "test-task"
    assert ("image" in received) is expects_image


def test_create_environment_keeps_env_type_for_plugin_backends(monkeypatch):
    """Unknown backends still receive their name for plugin provider lookup."""
    received = {}

    def build_plugin_env(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(backends, "_build_plugin_env", build_plugin_env)

    backends._create_environment(
        env_type="plugin-sandbox",
        image="test-image",
        cwd="/workspace",
        timeout=42,
        container_config={},
        task_id="test-task",
    )

    assert received["env_type"] == "plugin-sandbox"
    assert received["image"] == "test-image"
