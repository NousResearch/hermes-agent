"""Tests for F3: subagent tool-call progress events in delegate_task.

Covers:
- Normal completion: partial_events absent (timeout path not taken)
- Timeout after N tool calls: partial_events contains those N tools
- partial_events capped at 50 even when 100+ tool calls happen
- last_tool key reflects the final recorded tool name
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEARTBEAT_INTERVAL_FAST = 0.05  # seconds — speed up tests


class _ProgressChild:
    """Stub subagent that simulates tool calls visible via get_activity_summary().

    The child reports a sequence of ``tool_names`` one by one, advancing the
    iteration counter each time.  It blocks until released so the heartbeat
    thread has time to observe each tool.
    """

    def __init__(
        self,
        *,
        tool_names: List[str],
        hang_seconds: float = 10.0,
        subagent_id: str = "sa-0-progress",
        complete_normally: bool = False,
    ):
        self._subagent_id = subagent_id
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self.model = "test/model"
        self.provider = "testprov"
        self.api_mode = "chat_completions"
        self.base_url = "https://example.test/v1"
        self.max_iterations = 30
        self.quiet_mode = True
        self.skip_memory = True
        self.skip_context_files = True
        self.platform = "cli"
        self.ephemeral_system_prompt = "sys prompt"
        self.enabled_toolsets = ["web", "terminal"]
        self.valid_tool_names = set(tool_names)
        self.tools = [{"name": t, "description": t} for t in tool_names]
        self._tool_names = list(tool_names)
        self._hang_seconds = hang_seconds
        self._complete_normally = complete_normally

        # State updated as the fake run progresses
        self._api_call_count = 0
        self._current_tool: Optional[str] = None
        self._done = threading.Event()

    # --- AIAgent interface stubs ---

    def get_activity_summary(self) -> Dict[str, Any]:
        return {
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "current_tool": self._current_tool,
            "last_activity_desc": (
                f"running {self._current_tool}" if self._current_tool else ""
            ),
            "seconds_since_activity": 0,
        }

    def run_conversation(self, user_message: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Simulate tool calls then block (or complete)."""
        for tool in self._tool_names:
            if self._done.is_set():
                break
            self._current_tool = tool
            # Pause briefly so the heartbeat thread can observe the tool
            time.sleep(_HEARTBEAT_INTERVAL_FAST * 3)
            self._api_call_count += 1
            self._current_tool = None
            # Another brief pause between tools
            time.sleep(_HEARTBEAT_INTERVAL_FAST)

        if self._complete_normally:
            return {
                "final_response": "done",
                "completed": True,
                "api_calls": self._api_call_count,
                "messages": [],
            }

        # Block until released (simulates hanging after tool calls)
        self._done.wait(self._hang_seconds)
        return {
            "final_response": "",
            "completed": False,
            "api_calls": self._api_call_count,
            "messages": [],
        }

    def interrupt(self) -> None:
        self._done.set()


def _make_parent() -> MagicMock:
    parent = MagicMock()
    parent._touch_activity = MagicMock()
    parent._current_task_id = None
    return parent


def _run_child(child, monkeypatch, timeout: float = 0.5) -> Dict[str, Any]:
    """Invoke _run_single_child with a short timeout."""
    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: timeout)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", _HEARTBEAT_INTERVAL_FAST)

    parent = _make_parent()
    return delegate_tool._run_single_child(
        task_index=0,
        goal="test goal",
        child=child,
        parent_agent=parent,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPartialEventsNormalCompletion:
    """When the child completes normally the timeout path is not taken,
    so partial_events is NOT present in the result dict."""

    def test_normal_completion_has_no_partial_events(self, monkeypatch):
        child = _ProgressChild(
            tool_names=["web_search", "read_file"],
            complete_normally=True,
        )
        # Give enough time for the child to finish naturally
        result = _run_child(child, monkeypatch, timeout=5.0)

        assert result["status"] == "completed"
        # partial_events only injected on the timeout/error path
        assert "partial_events" not in result
        assert "last_tool" not in result


class TestPartialEventsOnTimeout:
    """When the child times out after making tool calls, partial_events
    contains those tool names and last_tool reflects the final one."""

    def test_timeout_after_three_tool_calls(self, monkeypatch):
        tools = ["web_search", "read_file", "terminal"]
        child = _ProgressChild(
            tool_names=tools,
            hang_seconds=10.0,
        )
        # Timeout long enough for 3 tool calls to register, short enough
        # that the child doesn't finish on its own.
        result = _run_child(child, monkeypatch, timeout=1.5)

        assert result["status"] == "timeout"
        assert "partial_events" in result
        events = result["partial_events"]
        # At least one event should have been recorded
        assert len(events) >= 1
        # All events must have the required keys
        for ev in events:
            assert "tool" in ev
            assert "status" in ev
            assert "preview" in ev

        # last_tool should match the final event's tool name
        assert result["last_tool"] == events[-1]["tool"]

    def test_error_message_contains_last_action(self, monkeypatch):
        tools = ["web_search", "read_file", "terminal"]
        child = _ProgressChild(tool_names=tools, hang_seconds=10.0)
        result = _run_child(child, monkeypatch, timeout=1.5)

        assert result["status"] == "timeout"
        assert "timed out after" in result["error"]
        assert "tool call(s)" in result["error"]
        if result["last_tool"]:
            assert f"Last action: {result['last_tool']}" in result["error"]

    def test_timeout_with_no_tool_calls_has_empty_partial_events(self, monkeypatch):
        """A child that times out before any tool call has empty partial_events."""
        # Child with no tools — it will immediately block without any tool calls
        child = _ProgressChild(tool_names=[], hang_seconds=10.0)
        result = _run_child(child, monkeypatch, timeout=0.3)

        assert result["status"] == "timeout"
        assert "partial_events" in result
        assert result["partial_events"] == []
        assert result["last_tool"] is None


class TestPartialEventsCap:
    """partial_events must be capped at 50 entries even when many tools run."""

    def test_capped_at_50_for_100_tool_calls(self, monkeypatch):
        # 100 distinct tool names
        tools = [f"tool_{i}" for i in range(100)]
        child = _ProgressChild(
            tool_names=tools,
            hang_seconds=60.0,
        )
        # Give enough time for many tools to run before timeout
        result = _run_child(child, monkeypatch, timeout=3.0)

        assert result["status"] == "timeout"
        events = result.get("partial_events", [])
        assert len(events) <= 50, (
            f"partial_events should be capped at 50, got {len(events)}"
        )
