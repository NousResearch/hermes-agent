import json
from unittest.mock import Mock

from tools import terminal_tool
from tools.approval import check_all_command_guards


def test_approval_guard_returns_decision_packet_for_git_push():
    result = check_all_command_guards("git push origin main", "local")

    assert result["approved"] is False
    assert result["status"] == "needs_chad"
    assert result["decision_packet"]["status"] == "NEEDS_CHAD"
    assert "git push" in result["message"]


def test_approval_guard_allows_read_only_command():
    result = check_all_command_guards("git status --short", "local")

    assert result["approved"] is True


def test_destructive_root_delete_still_hits_hardline_block():
    result = check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_terminal_tool_returns_decision_packet_before_starting_environment(monkeypatch):
    cleanup = Mock(side_effect=AssertionError("environment cleanup thread should not start"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", cleanup)

    data = json.loads(terminal_tool.terminal_tool("git commit -m test"))

    assert data["status"] == "needs_chad"
    assert data["decision_packet"]["status"] == "NEEDS_CHAD"
    assert "git commit" in data["error"]
    cleanup.assert_not_called()
