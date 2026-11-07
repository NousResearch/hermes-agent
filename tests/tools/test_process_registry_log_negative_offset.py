"""Tests for read_log() negative offsets in tools/process_registry.py (issue #146914).

read_log(offset=-N) means "the last N lines": the old lines[offset:offset + limit]
turned offset=-80/limit=80 into lines[-80:0], which is always empty, so any negative
offset returned no output. New tests live in their own file because the
test_process_registry.py line ratchet is over target on main."""

import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(sid: str, output: str) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    return ProcessSession(
        id=sid,
        command="echo hello",
        task_id="t1",
        started_at=time.time(),
        exited=False,
        exit_code=None,
        output_buffer=output,
    )


class TestReadLogNegativeOffset:
    def test_positive_offset_still_reads_forward(self, registry):
        """Positive offsets read forward from the head and may reach the log end."""
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session("proc_neg_pos", lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=90, limit=200)
        assert result["output"].splitlines() == [f"line {i}" for i in range(90, 100)]
        assert result["showing"] == "10 lines"

    def test_negative_offset_counts_from_end(self, registry):
        """A negative offset counts from the end of the log (#146914).

        The old lines[offset:offset + limit] turned offset=-80/limit=80 into
        lines[-80:0], which is always empty regardless of log length."""
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session("proc_neg_tail", lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=-80, limit=80)
        assert result["output"].splitlines() == [f"line {i}" for i in range(20, 100)]
        assert result["showing"] == "80 lines"

    def test_negative_offset_short_log_reports_issue_repro(self, registry):
        """The exact shape reported in #146914: one-line log, offset=-200, limit=200."""
        s = _make_session("proc_neg_one", "only line")
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=-200, limit=200)
        assert result["output"] == "only line"
        assert result["total_lines"] == 1
        assert result["showing"] == "1 lines"

    def test_negative_offset_with_smaller_limit(self, registry):
        """Negative offset with a limit smaller than the tail window: start -offset
        lines from the end, then cap the read at limit."""
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session("proc_neg_page", lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=-80, limit=5)
        assert result["output"].splitlines() == [f"line {i}" for i in range(20, 25)]
        assert result["showing"] == "5 lines"
        # A log shorter than the window: the whole log is the window, capped at limit
        # (the old lines[offset:offset + limit] returned lines[-10:-7], empty here too).
        short = _make_session("proc_neg_short", "a\nb\nc\nd\ne")
        registry._running[short.id] = short
        result = registry.read_log(short.id, offset=-10, limit=3)
        assert result["output"].splitlines() == ["a", "b", "c"]
        assert result["total_lines"] == 5
