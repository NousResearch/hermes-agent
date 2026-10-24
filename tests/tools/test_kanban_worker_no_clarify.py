"""A dispatcher-spawned Kanban worker is headless: clarify can only come back
"undelivered", so the question is lost. Workers ask through kanban_block (#32167)."""

from model_tools import _select_tool_names


def test_kanban_worker_does_not_get_clarify(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert "clarify" not in _select_tool_names(["hermes-cli"], None, quiet_mode=True)


def test_chat_session_keeps_clarify(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert "clarify" in _select_tool_names(["hermes-cli"], None, quiet_mode=True)
