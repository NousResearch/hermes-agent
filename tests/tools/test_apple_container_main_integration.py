"""Behavioral wiring contracts after integrating the shared backend modules."""

import json
from types import SimpleNamespace


def test_apple_config_and_override_reach_shared_factory(monkeypatch):
    from tools.terminal_tool_config import _is_container_backend

    assert _is_container_backend("apple_container") is True

    from tools import terminal_tool as terminal
    from tools import terminal_tool_backends as backends
    from tools.environments import apple_container as apple

    monkeypatch.setattr(terminal, "_ensure_terminal_env_bridged", lambda: None)
    monkeypatch.setenv("TERMINAL_ENV", "apple_container")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_IMAGE", "fixture:configured")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_VOLUMES", "[]")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_EXTRA_ARGS", json.dumps(["--network", "none"]))
    monkeypatch.setenv("TERMINAL_CONTAINER_CPU", "2")
    monkeypatch.setenv("TERMINAL_CONTAINER_MEMORY", "2048")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    config = terminal._get_env_config()
    config["fixture_plugin_key"] = "preserved"
    shaped = backends._container_config_from_config(config)
    assert shaped["apple_container_extra_args"] == ["--network", "none"]
    assert shaped["fixture_plugin_key"] == "preserved"
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(apple, "AppleContainerEnvironment", construct)
    image = terminal._select_image(
        "apple_container", {"apple_container_image": "fixture:override"}, config
    )
    backends._create_environment(
        "apple_container", image, "/workspace", 30,
        container_config=shaped, task_id="fixture-task",
    )
    assert captured["image"] == "fixture:override"
    assert captured["cpu"] == 2
    assert captured["memory"] == 2048
    assert captured["persistent_filesystem"] is True
    assert captured["extra_args"] == ["--network", "none"]
    assert captured["task_id"] == "fixture-task"
