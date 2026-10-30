import json
from unittest.mock import MagicMock, patch


def test_write_file_blocks_protected_root(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
    from tools.file_tools import write_file_tool

    with patch("tools.file_tools._get_file_ops") as get_ops:
        result = json.loads(write_file_tool(str(root / "agent.py"), "data"))
    assert "protected checkout" in result["error"]
    get_ops.assert_not_called()


def test_write_file_allows_linked_worktree(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    worktree = root / ".worktrees" / "task"
    worktree.mkdir(parents=True)
    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(root))
    from tools.file_tools import write_file_tool

    ops = MagicMock()
    result_obj = MagicMock()
    result_obj.to_dict.return_value = {"status": "ok"}
    ops.write_file.return_value = result_obj
    with patch("tools.file_tools._get_file_ops", return_value=ops):
        result = json.loads(write_file_tool(str(worktree / "agent.py"), "data"))
    assert result["status"] == "ok"
    ops.write_file.assert_called_once()
