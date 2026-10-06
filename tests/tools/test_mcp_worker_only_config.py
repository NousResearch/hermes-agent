"""Task-bound MCP servers must not connect during gateway/profile startup."""

from agent.delegation_context import non_dispatcher_owned_context
from tools.mcp_tool_config import _load_mcp_config


def test_worker_only_server_requires_owned_kanban_task(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n"
        "  task-only:\n"
        "    command: /usr/bin/true\n"
        "    worker_only: true\n"
        "  ordinary:\n"
        "    command: /usr/bin/true\n"
    )

    assert "ordinary" in _load_mcp_config()
    assert "task-only" not in _load_mcp_config()

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    assert "task-only" in _load_mcp_config()
    with non_dispatcher_owned_context():
        assert "task-only" not in _load_mcp_config()


def test_worker_only_keeps_native_precedence_over_portable_server(monkeypatch, tmp_path):
    from tools import mcp_tool_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        "mcp_servers:\n"
        "  task-only:\n"
        "    command: /usr/bin/true\n"
        "    worker_only: true\n"
    )
    monkeypatch.setattr(
        mcp_tool_config, "_portable_mcp_servers",
        lambda servers: servers.setdefault("task-only", {"command": "/usr/bin/false"}),
    )

    assert "task-only" not in _load_mcp_config()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    assert _load_mcp_config()["task-only"]["command"] == "/usr/bin/true"

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    config.write_text(config.read_text().replace("worker_only: true", 'worker_only: "true"'))
    assert "task-only" not in _load_mcp_config()
