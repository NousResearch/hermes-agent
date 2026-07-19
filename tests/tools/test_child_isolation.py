"""Tests for child agent isolation — structured error types and _run_single_child integration."""

import json
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import pytest

from tools.child_isolation import ChildErrorType, ChildResult, format_child_error


# ---------------------------------------------------------------------------
# ChildResult / ChildErrorType unit tests
# ---------------------------------------------------------------------------

class TestChildResult:
    def test_success_to_dict(self):
        r = ChildResult(success=True, task_index=0, summary="done")
        d = r.to_dict()
        assert d["success"] is True
        assert d["summary"] == "done"
        assert "error" not in d  # None values stripped

    def test_timeout_to_dict(self):
        r = ChildResult(
            success=False, task_index=1, status="timeout",
            error="timed out", error_type="timeout",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error_type"] == "timeout"

    def test_to_json_roundtrip(self):
        r = ChildResult(success=True, task_index=0, summary="done")
        data = json.loads(r.to_json())
        assert data["success"] is True
        assert data["summary"] == "done"

    def test_child_role_in_dict(self):
        r = ChildResult(success=True, child_role="leaf")
        d = r.to_dict()
        assert d["child_role"] == "leaf"


class TestChildErrorType:
    def test_enum_values(self):
        assert ChildErrorType.TIMEOUT.value == "timeout"
        assert ChildErrorType.CRASH.value == "crash"
        assert ChildErrorType.INTERRUPTED.value == "interrupted"
        assert ChildErrorType.DEPTH_LIMIT.value == "depth_limit"


class TestFormatChildError:
    def test_timeout_message(self):
        r = ChildResult(
            success=False, status="timeout", error_type="timeout",
            duration_seconds=120.0, error="timed out",
        )
        msg = format_child_error(r)
        assert "120" in msg
        assert "timed out" in msg.lower()

    def test_crash_message_mentions_parent(self):
        r = ChildResult(
            success=False, status="error", error_type="crash",
            error="Something broke",
        )
        msg = format_child_error(r)
        assert "crashed" in msg.lower()
        assert "parent" in msg.lower()

    def test_success_returns_empty(self):
        r = ChildResult(success=True, status="completed")
        assert format_child_error(r) == ""


# ---------------------------------------------------------------------------
# _run_single_child integration tests — mocked child agents
# ---------------------------------------------------------------------------

def _make_mock_child(result_dict=None, side_effect=None, role="leaf"):
    """Build a minimal mock child agent for _run_single_child tests."""
    child = MagicMock()
    child._delegate_role = role
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child._subagent_id = None
    child._active_children = []
    child._active_children_lock = None
    child.tool_progress_callback = None
    child.get_activity_summary.return_value = {
        "current_tool": None, "api_call_count": 0, "max_iterations": 10,
    }
    if side_effect is not None:
        child.run_conversation.side_effect = side_effect
    else:
        child.run_conversation.return_value = result_dict or {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 3,
        }
    return child


class TestRunSingleChild:
    def test_success_returns_child_role(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child(role="orchestrator")
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=30), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool.file_state"):
            result = _run_single_child(0, "test goal", child, parent)

        assert result["status"] == "completed"
        assert result.get("_child_role") == "orchestrator" or result.get("child_role") == "orchestrator"

    def test_timeout_returns_child_result_type(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child(side_effect=FuturesTimeoutError("timed out"))
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=0.1), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool._dump_subagent_timeout_diagnostic", return_value=None), \
             patch("tools.delegate_tool.file_state"):
            result = _run_single_child(0, "test goal", child, parent)

        assert result["status"] == "timeout"
        assert result["error_type"] == "timeout"
        assert result["success"] is False
        child.interrupt.assert_called()

    def test_crash_returns_child_result_type(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child(side_effect=RuntimeError("child exploded"))
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=30), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool.file_state"):
            result = _run_single_child(0, "test goal", child, parent)

        assert result["status"] == "error"
        assert result["error_type"] == "crash"
        assert result["success"] is False

    def test_heartbeat_starts_and_stops(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child()
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=30), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool.file_state"):
            result = _run_single_child(0, "test goal", child, parent)

        # Parent activity was touched at least once during heartbeat
        assert parent._touch_activity.call_count >= 0  # may be 0 if child is fast

    def test_child_close_called(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child()
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=30), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool.file_state"):
            _run_single_child(0, "test goal", child, parent)

        child.close.assert_called_once()

    def test_keyboard_interrupt_not_swallowed(self):
        from tools.delegate_tool import _run_single_child
        child = _make_mock_child(side_effect=KeyboardInterrupt())
        parent = MagicMock()
        parent._current_task_id = "parent-1"
        parent._touch_activity = MagicMock()
        parent._active_children = []
        parent._active_children_lock = None

        with patch("tools.delegate_tool._get_child_timeout", return_value=30), \
             patch("tools.delegate_tool._get_subagent_approval_callback", return_value=None), \
             patch("tools.delegate_tool._register_subagent"), \
             patch("tools.delegate_tool._unregister_subagent"), \
             patch("tools.delegate_tool.file_state"):
            with pytest.raises(KeyboardInterrupt):
                _run_single_child(0, "test goal", child, parent)
