from types import SimpleNamespace


def test_task_override_selects_isolated_backend(monkeypatch):
    from tools import terminal_tool, terminal_tool_backends

    created = {}
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {
        "env_type": "local", "cwd": ".", "timeout": 30,
    })

    def create(**kwargs):
        created.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(terminal_tool_backends, "_create_environment", create)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    terminal_tool.register_task_env_overrides("cron-session", {
        "env_type": "docker",
        "docker_image": "fleet:test",
        "docker_network": False,
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,
        "container_persistent": False,
        "docker_persist_across_processes": False,
        "cwd": "/workspace",
    })
    try:
        terminal_tool.ensure_task_env("cron-session")
        assert created["env_type"] == "docker"
        assert created["image"] == "fleet:test"
        assert created["host_cwd"] is None
        assert created["container_config"]["docker_network"] is False
        assert created["container_config"]["docker_volumes"] == []
    finally:
        terminal_tool.clear_task_env_overrides("cron-session")
        terminal_tool._active_environments.pop("cron-session", None)
