"""
Regression tests for kanban_create skills validation (issue #22921).

kanban_create previously accepted any string array in the skills field,
including toolset names like "web", "browser", "terminal". When the
dispatcher spawned a worker with --skills web browser, it immediately
crashed with "Error: Unknown skill(s): web, browser".
"""

import json
from unittest.mock import patch, MagicMock


class TestKanbanCreateSkillsValidation:
    """Tests that kanban_create rejects toolset names in skills field."""

    @patch("tools.kanban_tools._connect")
    def test_rejects_single_toolset_name(self, mock_connect):
        """Creating a task with skills=[\"web\"] should fail fast."""
        from tools.kanban_tools import _handle_create

        result = _handle_create({
            "title": "Research competitors",
            "assignee": "researcher",
            "skills": ["web"],
        })
        data = json.loads(result)
        assert data.get("success") is False
        assert "web" in data.get("error", "")
        assert "toolset" in data.get("error", "").lower()
        mock_connect.assert_not_called()

    @patch("tools.kanban_tools._connect")
    def test_rejects_multiple_toolset_names(self, mock_connect):
        """Creating a task with skills=[\"web\", \"browser\"] should fail fast."""
        from tools.kanban_tools import _handle_create

        result = _handle_create({
            "title": "Research competitors",
            "assignee": "researcher",
            "skills": ["web", "browser"],
        })
        data = json.loads(result)
        assert data.get("success") is False
        assert "web" in data.get("error", "")
        assert "browser" in data.get("error", "")
        mock_connect.assert_not_called()

    @patch("tools.kanban_tools._connect")
    def test_rejects_mixed_toolset_and_skill_names(self, mock_connect):
        """If any toolset name is present, the entire request should fail."""
        from tools.kanban_tools import _handle_create

        result = _handle_create({
            "title": "Research competitors",
            "assignee": "researcher",
            "skills": ["translation", "web"],  # translation is valid, web is not
        })
        data = json.loads(result)
        assert data.get("success") is False
        assert "web" in data.get("error", "")
        mock_connect.assert_not_called()

    @patch("tools.kanban_tools._connect")
    def test_accepts_valid_skill_names(self, mock_connect):
        """Valid skill names should be accepted and passed to create_task."""
        from tools.kanban_tools import _handle_create

        mock_kb = MagicMock()
        mock_conn = MagicMock()
        mock_kb.create_task.return_value = "t_123"
        mock_kb.get_task.return_value = MagicMock(status="todo")
        mock_connect.return_value = (mock_kb, mock_conn)

        result = _handle_create({
            "title": "Translate docs",
            "assignee": "translator",
            "skills": ["translation", "github-code-review"],
        })
        data = json.loads(result)
        assert data.get("success") is True
        assert data.get("task_id") == "t_123"

    @patch("tools.kanban_tools._connect")
    def test_accepts_empty_skills(self, mock_connect):
        """Empty skills list should be accepted."""
        from tools.kanban_tools import _handle_create

        mock_kb = MagicMock()
        mock_conn = MagicMock()
        mock_kb.create_task.return_value = "t_456"
        mock_kb.get_task.return_value = MagicMock(status="todo")
        mock_connect.return_value = (mock_kb, mock_conn)

        result = _handle_create({
            "title": "Simple task",
            "assignee": "worker",
            "skills": [],
        })
        data = json.loads(result)
        assert data.get("success") is True

    @patch("tools.kanban_tools._connect")
    def test_rejects_terminal_toolset(self, mock_connect):
        """\"terminal\" is a toolset, not a skill."""
        from tools.kanban_tools import _handle_create

        result = _handle_create({
            "title": "Run scripts",
            "assignee": "devops",
            "skills": ["terminal"],
        })
        data = json.loads(result)
        assert data.get("success") is False
        assert "terminal" in data.get("error", "")

    @patch("tools.kanban_tools._connect")
    def test_rejects_file_toolset(self, mock_connect):
        """\"file\" is a toolset, not a skill."""
        from tools.kanban_tools import _handle_create

        result = _handle_create({
            "title": "File ops",
            "assignee": "worker",
            "skills": ["file"],
        })
        data = json.loads(result)
        assert data.get("success") is False
        assert "file" in data.get("error", "")


if __name__ == "__main__":
    import sys

    test_class = TestKanbanCreateSkillsValidation()
    methods = [m for m in dir(test_class) if m.startswith("test_")]
    passed = 0
    failed = 0

    for method_name in methods:
        try:
            getattr(test_class, method_name)()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
