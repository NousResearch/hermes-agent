"""A plugin terminal backend gets its per-task image override (``<backend>_image``, what batch runs
register for the active backend), and that override marks the task as an isolated rollout like a
built-in backend's image override does."""

from tools import terminal_tool as tt


def test_plugin_backend_reads_its_own_per_task_image_override():
    overrides = {"docker_image": "python:3.12", "acmebox_image": "python:3.12"}
    assert tt._select_image("acmebox", overrides, {}) == "python:3.12"
    assert tt._select_image("acmebox", {"docker_image": "python:3.12"}, {}) == ""
    assert tt._select_image("local", overrides, {}) == ""
    assert tt._select_image("docker", overrides, {"docker_image": "cfg"}) == "python:3.12"


def test_plugin_image_override_marks_an_isolated_rollout(monkeypatch):
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    tt.register_task_env_overrides("rollout-1", {"acmebox_image": "python:3.12"})
    tt.register_task_env_overrides("cwd-only", {"cwd": "/work"})
    assert tt._has_isolation_overrides("rollout-1") is True
    assert tt._has_isolation_overrides("cwd-only") is False
