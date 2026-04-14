"""Tests for session-scoped taint tracking (tools/taint_context.py)."""

import threading

import pytest

from tools.taint_context import (
    TaintSource,
    TaintState,
    clear_taint,
    get_taint,
    is_tainted,
    mark_tainted,
    _lock,
    _session_taint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_taint_state():
    """Ensure a pristine global taint registry for every test."""
    with _lock:
        _session_taint.clear()
    yield
    with _lock:
        _session_taint.clear()


# ---------------------------------------------------------------------------
# TaintState unit tests
# ---------------------------------------------------------------------------

class TestTaintState:
    def test_mark_single_source(self):
        state = TaintState()
        state.mark(TaintSource.WEB_FETCH, "web_extract")
        assert state.tainted is True
        assert state.sources == [TaintSource.WEB_FETCH]
        assert state.first_taint_tool == "web_extract"

    def test_mark_multiple_sources(self):
        state = TaintState()
        state.mark(TaintSource.WEB_FETCH, "web_extract")
        state.mark(TaintSource.MCP_RESULT, "mcp_github")
        state.mark(TaintSource.INBOUND_MSG, "gateway_inbound")
        assert state.sources == [
            TaintSource.WEB_FETCH,
            TaintSource.MCP_RESULT,
            TaintSource.INBOUND_MSG,
        ]
        # first_taint_tool should not change after the first mark
        assert state.first_taint_tool == "web_extract"

    def test_mark_duplicate_source_not_appended(self):
        state = TaintState()
        state.mark(TaintSource.WEB_SEARCH, "web_search")
        state.mark(TaintSource.WEB_SEARCH, "web_search")
        assert state.sources == [TaintSource.WEB_SEARCH]

    def test_summary_clean(self):
        state = TaintState()
        assert state.summary() == "clean"

    def test_summary_tainted(self):
        state = TaintState()
        state.mark(TaintSource.WEB_FETCH, "web_extract")
        state.mark(TaintSource.SUBAGENT, "delegate")
        summary = state.summary()
        assert "tainted" in summary
        assert "web_fetch" in summary
        assert "subagent" in summary
        assert "web_extract" in summary


# ---------------------------------------------------------------------------
# Module-level function tests
# ---------------------------------------------------------------------------

class TestModuleFunctions:
    def test_get_taint_creates_new_state(self):
        state = get_taint("session-abc")
        assert isinstance(state, TaintState)
        assert state.tainted is False

    def test_get_taint_returns_same_state(self):
        s1 = get_taint("session-xyz")
        s2 = get_taint("session-xyz")
        assert s1 is s2

    def test_mark_tainted_propagates(self):
        mark_tainted("sess-1", TaintSource.WEB_SEARCH, "web_search")
        state = get_taint("sess-1")
        assert state.tainted is True
        assert TaintSource.WEB_SEARCH in state.sources
        assert state.first_taint_tool == "web_search"

    def test_is_tainted_false_for_clean(self):
        assert is_tainted("never-seen") is False

    def test_is_tainted_true_after_marking(self):
        mark_tainted("sess-2", TaintSource.MCP_RESULT, "mcp_tool")
        assert is_tainted("sess-2") is True

    def test_clear_taint_resets_state(self):
        mark_tainted("sess-3", TaintSource.INBOUND_MSG, "gateway_inbound")
        assert is_tainted("sess-3") is True
        clear_taint("sess-3")
        assert is_tainted("sess-3") is False
        # After clearing, a fresh state should be returned
        state = get_taint("sess-3")
        assert state.tainted is False
        assert state.sources == []

    def test_clear_taint_no_op_for_unknown_session(self):
        # Should not raise
        clear_taint("nonexistent-session")


# ---------------------------------------------------------------------------
# Isolation tests
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_taint_does_not_leak_between_sessions(self):
        mark_tainted("sess-a", TaintSource.WEB_FETCH, "web_extract")
        assert is_tainted("sess-a") is True
        assert is_tainted("sess-b") is False

    def test_independent_sources_per_session(self):
        mark_tainted("sess-x", TaintSource.WEB_SEARCH, "web_search")
        mark_tainted("sess-y", TaintSource.MCP_RESULT, "mcp_tool")
        state_x = get_taint("sess-x")
        state_y = get_taint("sess-y")
        assert state_x.sources == [TaintSource.WEB_SEARCH]
        assert state_y.sources == [TaintSource.MCP_RESULT]


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_marks_on_same_session(self):
        """Multiple threads marking the same session should not lose data."""
        sources = [
            (TaintSource.WEB_FETCH, "web_extract"),
            (TaintSource.WEB_SEARCH, "web_search"),
            (TaintSource.MCP_RESULT, "mcp_tool"),
            (TaintSource.INBOUND_MSG, "gateway_inbound"),
            (TaintSource.SUBAGENT, "delegate"),
        ]
        errors = []

        def mark_one(source, tool):
            try:
                mark_tainted("shared-session", source, tool)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=mark_one, args=(src, tool))
            for src, tool in sources
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"
        state = get_taint("shared-session")
        assert state.tainted is True
        # All five sources should be recorded (order may vary due to threading)
        assert set(state.sources) == {s for s, _ in sources}

    def test_concurrent_marks_on_different_sessions(self):
        """Concurrent marks on distinct sessions should not interfere."""
        errors = []

        def mark_session(sid, source, tool):
            try:
                mark_tainted(sid, source, tool)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=mark_session, args=(f"s-{i}", TaintSource.WEB_FETCH, "tool"))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        for i in range(20):
            assert is_tainted(f"s-{i}") is True
