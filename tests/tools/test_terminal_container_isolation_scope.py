import importlib


def _reload_terminal_tool(monkeypatch, isolation: str | None):
    if isolation is None:
        monkeypatch.delenv("TERMINAL_CONTAINER_ISOLATION", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_CONTAINER_ISOLATION", isolation)
    import tools.terminal_tool as terminal_tool

    return importlib.reload(terminal_tool)


def test_container_task_ids_share_default_sandbox_by_default(monkeypatch):
    terminal_tool = _reload_terminal_tool(monkeypatch, None)

    assert terminal_tool._resolve_container_task_id("session-a") == "default"
    assert terminal_tool._resolve_container_task_id("session-b") == "default"


def test_session_container_isolation_uses_session_task_id(monkeypatch):
    terminal_tool = _reload_terminal_tool(monkeypatch, "session")

    assert terminal_tool._resolve_container_task_id("session-a") == "session-a"
    assert terminal_tool._resolve_container_task_id("session-b") == "session-b"


def test_session_container_isolation_falls_back_to_context_session_id(monkeypatch):
    terminal_tool = _reload_terminal_tool(monkeypatch, "session")
    monkeypatch.setenv("HERMES_SESSION_ID", "gateway-session-123")

    assert terminal_tool._resolve_container_task_id(None) == "gateway-session-123"


def test_registered_task_env_overrides_remain_isolated_when_shared(monkeypatch):
    terminal_tool = _reload_terminal_tool(monkeypatch, "shared")
    terminal_tool.register_task_env_overrides("benchmark-task", {"docker_image": "example/image"})
    try:
        assert terminal_tool._resolve_container_task_id("benchmark-task") == "benchmark-task"
        assert terminal_tool._resolve_container_task_id("ordinary-task") == "default"
    finally:
        terminal_tool.clear_task_env_overrides("benchmark-task")
