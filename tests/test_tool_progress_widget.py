"""Tests for hermes_cli.tool_progress_widget — multi-lane tool progress tracker."""

import threading

import pytest

from hermes_cli.tool_progress_widget import (
    LANE_ACTIVE,
    LANE_DONE,
    LANE_ERROR,
    LANE_WAITING,
    ToolProgressTracker,
    build_progress_fragments,
)


class TestToolProgressTracker:
    def test_initial_state(self):
        tracker = ToolProgressTracker()
        assert not tracker.has_activity()
        assert tracker.get_active_lanes() == []
        assert tracker.get_turn_summary() == {}

    def test_tool_start_creates_lane(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        assert tracker.has_activity()
        lanes = tracker.get_active_lanes()
        assert len(lanes) == 1
        assert lanes[0].name == "terminal"
        assert lanes[0].status == LANE_ACTIVE
        assert lanes[0].tool_calls == 1

    def test_tool_complete_marks_done(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_complete("terminal")
        lanes = tracker.get_active_lanes()
        assert lanes[0].status == LANE_DONE

    def test_tool_complete_error(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_complete("terminal", success=False)
        lanes = tracker.get_active_lanes()
        assert lanes[0].status == LANE_ERROR

    def test_multiple_calls_increment_count(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_complete("terminal")
        tracker.on_tool_start("terminal", "ls -la")
        assert tracker.get_active_lanes()[0].tool_calls == 2

    def test_multiple_tools(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_start("write_file", "test.py")
        assert len(tracker.get_active_lanes()) == 2

    def test_turn_end_clears_all(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_start("write_file", "test.py")
        tracker.on_turn_end()
        assert not tracker.has_activity()
        assert tracker.get_turn_summary() == {}

    def test_turn_summary(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_start("terminal", "ls")
        tracker.on_tool_start("write_file", "test.py")
        summary = tracker.get_turn_summary()
        assert summary == {"terminal": 2, "write_file": 1}

    def test_thread_safety(self):
        tracker = ToolProgressTracker()
        errors = []

        def start_tools():
            try:
                for i in range(50):
                    tracker.on_tool_start(f"tool_{i % 5}", f"action_{i}")
            except Exception as e:
                errors.append(e)

        def complete_tools():
            try:
                for i in range(50):
                    tracker.on_tool_complete(f"tool_{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=start_tools),
            threading.Thread(target=complete_tools),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_invalidate_callback(self):
        tracker = ToolProgressTracker()
        calls = []
        tracker.set_invalidate_callback(lambda: calls.append(1))
        tracker.on_tool_start("terminal", "whoami")
        assert len(calls) >= 1


class TestBuildProgressFragments:
    def test_empty_when_no_activity(self):
        tracker = ToolProgressTracker()
        frags = build_progress_fragments(tracker, 80)
        assert frags == []

    def test_returns_fragments_with_activity(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        frags = build_progress_fragments(tracker, 80)
        assert len(frags) > 0
        # Should contain style/text tuples
        for style, text in frags:
            assert isinstance(style, str)
            assert isinstance(text, str)

    def test_contains_tool_name(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        frags = build_progress_fragments(tracker, 80)
        all_text = "".join(text for _, text in frags)
        assert "terminal" in all_text

    def test_narrow_width_does_not_crash(self):
        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        frags = build_progress_fragments(tracker, 30)
        assert len(frags) > 0
