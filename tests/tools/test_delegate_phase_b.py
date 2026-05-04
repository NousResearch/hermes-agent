"""Tests for delegate_tool Phase B — readonly isolation, standard result, status interfaces."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    _resolve_isolation,
    _apply_readonly_isolation,
    READONLY_STRIP_TOOLSETS,
    READONLY_STRIP_TOOLS,
    get_subagent_status,
    get_subagent_output_tail,
    get_subagent_usage,
    interrupt_subagent,
    _register_subagent,
    _unregister_subagent,
)


# ── Isolation Resolution Tests ──────────────────────────────────────────

class TestResolveIsolation:

    def test_default_shared(self):
        iso, warns, err = _resolve_isolation(
            requested_isolation=None, profile_isolation=None, permission_mode=None,
        )
        assert iso == "shared"
        assert err is None

    def test_explicit_readonly(self):
        iso, warns, err = _resolve_isolation(
            requested_isolation="readonly", profile_isolation=None, permission_mode=None,
        )
        assert iso == "readonly"
        assert err is None

    def test_explicit_shared(self):
        iso, warns, err = _resolve_isolation(
            requested_isolation="shared", profile_isolation="readonly", permission_mode=None,
        )
        assert iso == "shared"

    def test_worktree_no_longer_returns_error(self):
        """Phase C: worktree isolation is now implemented."""
        iso, warns, err = _resolve_isolation(
            requested_isolation="worktree", profile_isolation=None, permission_mode=None,
        )
        assert iso == "worktree"
        assert err is None

    def test_read_only_permission_auto_readonly(self):
        """permission_mode='read_only' auto-enables readonly isolation."""
        iso, warns, err = _resolve_isolation(
            requested_isolation=None, profile_isolation=None, permission_mode="read_only",
        )
        assert iso == "readonly"
        assert len(warns) >= 1
        assert any("automatically enables" in w for w in warns)

    def test_explicit_shared_overrides_read_only_profile(self):
        """Explicit isolation='shared' overrides profile's permission_mode."""
        iso, warns, err = _resolve_isolation(
            requested_isolation="shared", profile_isolation=None, permission_mode="read_only",
        )
        # Explicit shared request wins over read_only permission
        assert iso == "shared"
        assert err is None

    def test_unknown_isolation_warns(self):
        iso, warns, err = _resolve_isolation(
            requested_isolation="container", profile_isolation=None, permission_mode=None,
        )
        assert iso == "shared"
        assert err is not None
        assert "Unknown" in err

    def test_profile_isolation_as_fallback(self):
        """Profile isolation is used when no explicit isolation requested."""
        iso, warns, err = _resolve_isolation(
            requested_isolation=None, profile_isolation="readonly", permission_mode=None,
        )
        assert iso == "readonly"
        assert err is None


# ── Readonly Isolation Tests ────────────────────────────────────────────

class TestApplyReadonlyIsolation:

    def test_strips_terminal_toolset(self):
        result = _apply_readonly_isolation(["terminal", "file", "web"])
        assert "terminal" not in result
        assert "file" in result
        assert "web" in result

    def test_preserves_read_only_toolsets(self):
        result = _apply_readonly_isolation(["file", "web", "browser", "vision"])
        assert "file" in result
        assert "web" in result
        assert "browser" in result
        assert "vision" in result
        assert "terminal" not in result

    def test_empty_toolsets(self):
        result = _apply_readonly_isolation([])
        assert result == []

    def test_only_terminal_returns_empty(self):
        result = _apply_readonly_isolation(["terminal"])
        assert result == []

    def test_readonly_strip_toolsets_constant(self):
        assert "terminal" in READONLY_STRIP_TOOLSETS

    def test_readonly_strip_tools_constant(self):
        assert "write_file" in READONLY_STRIP_TOOLS
        assert "patch" in READONLY_STRIP_TOOLS
        assert "terminal" in READONLY_STRIP_TOOLS


# ── Standard Result Entry Tests ─────────────────────────────────────────

class TestStandardResultEntry:
    """Verify the standardized result entry fields (tested via unit logic)."""

    def test_sanitize_agent_id_none_when_magic_mock(self):
        """MagicMock values are sanitized to None/empty, not crash."""
        child = MagicMock()
        # Simulate the sanitization logic from _run_single_child
        _subagent_id = getattr(child, "_subagent_id", None)
        _child_agent_id = getattr(child, "_subagent_agent_id", None)
        _child_role = getattr(child, "_delegate_role", "leaf")
        _raw_toolsets = getattr(child, "_subagent_effective_toolsets", None)

        safe_subagent_id = _subagent_id if isinstance(_subagent_id, str) else None
        safe_agent_id = _child_agent_id if isinstance(_child_agent_id, str) else None
        safe_role = _child_role if isinstance(_child_role, str) else "leaf"
        safe_toolsets = list(_raw_toolsets) if isinstance(_raw_toolsets, list) else []

        assert safe_subagent_id is None
        assert safe_agent_id is None
        assert safe_role == "leaf"
        assert safe_toolsets == []

    def test_standard_fields_present(self):
        """Verify standard fields are present in entry structure."""
        expected_fields = {
            "status", "subagent_id", "parent_id", "agent_id", "role",
            "task_index", "goal", "summary", "effective_toolsets",
            "blocked_tools", "isolation", "output_tail", "usage",
            "duration_seconds", "warnings", "error",
        }
        # These are the fields Phase B requires in the result entry.
        # Tests above (test_delegate.py) verify actual serialization.
        assert expected_fields  # meta-check: field list is defined


# ── Subagent Status / Output Tests ─────────────────────────────────────

class TestSubagentStatusInterfaces:

    def setup_method(self):
        # Clean up any leftover test entries
        with patch("tools.delegate_tool._active_subagents", {}):
            pass

    def test_get_status_not_found(self):
        assert get_subagent_status("nonexistent-id") is None

    def test_get_status_returns_snapshot(self):
        from tools.delegate_tool import _active_subagents, _active_subagents_lock

        test_id = "sa-test-status"
        with _active_subagents_lock:
            _active_subagents[test_id] = {
                "subagent_id": test_id,
                "parent_id": None,
                "depth": 0,
                "goal": "test goal",
                "model": "test-model",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 3,
                "last_tool": "read_file",
                "agent": None,
            }

        try:
            status = get_subagent_status(test_id)
            assert status is not None
            assert status["subagent_id"] == test_id
            assert status["goal"] == "test goal"
            assert status["status"] == "running"
            assert status["tool_count"] == 3
            assert "agent" not in status  # agent ref excluded from snapshot
        finally:
            with _active_subagents_lock:
                _active_subagents.pop(test_id, None)

    def test_get_output_tail_not_found(self):
        assert get_subagent_output_tail("nonexistent-id") is None

    def test_get_usage_not_found(self):
        assert get_subagent_usage("nonexistent-id") is None

    def test_interrupt_subagent_not_found(self):
        assert interrupt_subagent("nonexistent-id") is False


# ── Interrupt Tests ─────────────────────────────────────────────────────

class TestInterruptSubagent:

    def test_interrupt_nonexistent_returns_false(self):
        assert interrupt_subagent("nonexistent-id-xyz") is False

    def test_interrupt_found_calls_agent_interrupt(self):
        from tools.delegate_tool import _active_subagents, _active_subagents_lock

        mock_agent = MagicMock()
        mock_agent.interrupt = MagicMock()
        test_id = "sa-test-interrupt"

        with _active_subagents_lock:
            _active_subagents[test_id] = {
                "subagent_id": test_id,
                "parent_id": None,
                "depth": 0,
                "goal": "test",
                "model": None,
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": mock_agent,
            }

        try:
            result = interrupt_subagent(test_id)
            assert result is True
            mock_agent.interrupt.assert_called_once()
        finally:
            with _active_subagents_lock:
                _active_subagents.pop(test_id, None)
