"""Routing regressions for terminal calls using Apple Container."""

from unittest.mock import Mock

import tools.environments.apple_container as apple
import tools.terminal_tool as terminal_tool


def _config() -> dict:
    return {
        "env_type": "apple_container",
        "apple_container_image": "python:3.12-slim",
        "apple_container_volumes": ["/host/data:/workspace/data:ro"],
        "apple_container_extra_args": ["--network", "none"],
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 42,
        "lifetime_seconds": 300,
        "container_cpu": 4,
        "container_memory": 6144,
        "container_disk": 51200,
        "container_persistent": True,
        "modal_mode": "auto",
        "vercel_runtime": "",
    }


def test_factory_passes_all_apple_container_settings(monkeypatch):
    fake = Mock(return_value=object())
    monkeypatch.setattr(apple, "AppleContainerEnvironment", fake)

    result = terminal_tool._create_environment(
        env_type="apple_container",
        image="python:3.12-slim",
        cwd="/workspace",
        timeout=42,
        container_config={
            "container_cpu": 4,
            "container_memory": 6144,
            "container_persistent": True,
            "apple_container_image": "python:3.12-slim",
            "apple_container_volumes": ["/host/data:/workspace/data:ro"],
            "apple_container_extra_args": ["--network", "none"],
        },
        task_id="task-1",
    )

    assert result is fake.return_value
    fake.assert_called_once_with(
        image="python:3.12-slim",
        cwd="/workspace",
        timeout=42,
        cpu=4,
        memory=6144,
        persistent_filesystem=True,
        task_id="task-1",
        volumes=["/host/data:/workspace/data:ro"],
        extra_args=["--network", "none"],
    )


def test_terminal_creation_passes_apple_specific_config(monkeypatch):
    captured = {}

    class DummyEnvironment:
        cwd = "/workspace"

        def execute(self, *args, **kwargs):
            return {"output": "", "exit_code": 0}

    def fake_create_environment(**kwargs):
        captured.update(kwargs)
        return DummyEnvironment()

    monkeypatch.setattr(terminal_tool, "_get_env_config", _config)
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})

    terminal_tool.terminal_tool(command="pwd", task_id="task-route")

    assert captured["image"] == "python:3.12-slim"
    assert captured["container_config"]["apple_container_image"] == "python:3.12-slim"
    assert captured["container_config"]["apple_container_volumes"] == [
        "/host/data:/workspace/data:ro"
    ]
    assert captured["container_config"]["apple_container_extra_args"] == [
        "--network", "none"
    ]
