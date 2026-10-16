"""Session-total delegation budget regressions for #52484."""

import json
import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task


def test_session_budget_reads_user_configuration(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "delegation:\n  max_children_per_session: 1\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    parent = _make_mock_parent()
    parent.session_id = "configured-budget"
    from tools.delegate_tool import cleanup_session_budget

    try:
        with patch("tools.delegate_tool._run_single_child") as run_child:
            run_child.return_value = TestSessionChildrenCap._completed_result()
            first = json.loads(delegate_task(goal="First configured task", parent_agent=parent))
            second = json.loads(delegate_task(goal="Second configured task", parent_agent=parent))
        assert "results" in first
        assert "Session children cap reached (1/1" in second["error"]
        assert run_child.call_count == 1
    finally:
        cleanup_session_budget(parent.session_id)


class TestDelegateRequirements(unittest.TestCase):
    def test_schema_description_advertises_session_budget(self):
        from tools.delegate_tool import _build_dynamic_schema_overrides

        description = _build_dynamic_schema_overrides()["description"]

        self.assertIn("SESSION BUDGET", description)


class TestSessionChildrenCap(unittest.TestCase):
    def setUp(self):
        from tools.delegate_tool import _session_children_counts

        _session_children_counts.clear()

    def tearDown(self):
        from tools.delegate_tool import _session_children_counts

        _session_children_counts.clear()

    @staticmethod
    def _completed_result():
        return {
            "task_index": 0,
            "status": "completed",
            "summary": "Done",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }

    @patch("tools.delegate_tool._run_single_child")
    def test_repeated_calls_stop_at_session_cap(self, mock_run):
        mock_run.return_value = self._completed_result()
        parent = _make_mock_parent()
        parent.session_id = "session-cap-test"

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"max_children_per_session": 3},
        ):
            for i in range(3):
                result = json.loads(
                    delegate_task(goal=f"Task {i}", parent_agent=parent)
                )
                self.assertIn("results", result)

            rejected = json.loads(
                delegate_task(goal="Task over cap", parent_agent=parent)
            )

        self.assertIn("error", rejected)
        self.assertIn("session", rejected["error"].lower())
        self.assertIn("3", rejected["error"])
        self.assertEqual(mock_run.call_count, 3)

    @patch("tools.delegate_tool._run_single_child")
    def test_completed_children_do_not_release_slots(self, mock_run):
        mock_run.return_value = self._completed_result()
        parent = _make_mock_parent()
        parent.session_id = "session-total-test"

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"max_children_per_session": 2},
        ):
            for i in range(2):
                self.assertIn(
                    "results",
                    json.loads(delegate_task(goal=f"Task {i}", parent_agent=parent)),
                )
            first_rejection = json.loads(
                delegate_task(goal="Over cap", parent_agent=parent)
            )
            second_rejection = json.loads(
                delegate_task(goal="After finish", parent_agent=parent)
            )

        self.assertIn("error", first_rejection)
        self.assertIn("error", second_rejection)

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_consumes_one_slot_per_child(self, mock_run):
        mock_run.return_value = self._completed_result()
        parent = _make_mock_parent()
        parent.session_id = "session-batch-test"

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"max_children_per_session": 3},
        ):
            batch = json.loads(
                delegate_task(
                    tasks=[{"goal": "First batch task A"}, {"goal": "Second batch task B"}],
                    parent_agent=parent,
                )
            )
            final_slot = json.loads(
                delegate_task(goal="Single after batch", parent_agent=parent)
            )
            rejected = json.loads(
                delegate_task(goal="Over cap", parent_agent=parent)
            )

        self.assertIn("results", batch)
        self.assertIn("results", final_slot)
        self.assertIn("error", rejected)

    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._run_single_child")
    def test_build_failure_releases_complete_batch(self, mock_run, mock_build):
        mock_run.return_value = self._completed_result()
        mock_build.side_effect = [MagicMock(), RuntimeError("child build failed")]
        parent = _make_mock_parent()
        parent.session_id = "session-build-fail-test"

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"max_children_per_session": 2},
        ):
            with self.assertRaises(RuntimeError):
                delegate_task(
                    tasks=[{"goal": "First build failure task"}, {"goal": "Second build failure task"}],
                    parent_agent=parent,
                )
            mock_run.assert_not_called()

            mock_build.side_effect = None
            mock_build.return_value = MagicMock()
            retry = json.loads(
                delegate_task(
                    tasks=[{"goal": "Retry first child task"}, {"goal": "Retry second child task"}],
                    parent_agent=parent,
                )
            )

        self.assertEqual(len(retry["results"]), 2)

    @patch("tools.delegate_tool._run_single_child")
    def test_cleanup_removes_session_entry(self, mock_run):
        from tools.delegate_tool import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        mock_run.return_value = self._completed_result()
        parent = _make_mock_parent()
        parent.session_id = "session-cleanup-test"

        with patch(
            "tools.delegate_tool._load_config",
            return_value={"max_children_per_session": 5},
        ):
            result = json.loads(delegate_task(goal="Task", parent_agent=parent))

        self.assertIn("results", result)
        with _session_children_lock:
            self.assertIn(parent.session_id, _session_children_counts)
        cleanup_session_budget(parent.session_id)
        with _session_children_lock:
            self.assertNotIn(parent.session_id, _session_children_counts)
        cleanup_session_budget("never-existed")
        cleanup_session_budget("")
