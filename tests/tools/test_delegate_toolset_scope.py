"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace

from tools.delegate_tool import _strip_blocked_tools, _emit_parent_console


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped


    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []


class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")
        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""


    def test_non_callable_safe_print_is_ignored(self, capsys):
        """Defensive: if _safe_print is set but not callable, fall back."""
        parent = SimpleNamespace(_safe_print="not-a-function")
        _emit_parent_console(parent, "  ✓ [3/3] non-callable guard")
        captured = capsys.readouterr()
        assert "non-callable guard" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 (toolset starvation) regression — exercises the REAL _build_child_agent
# (not the inline simulation the legacy TestToolsetIntersection above uses).
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_child_agent, _strip_blocked_tools


def _mock_parent(enabled_toolsets):
    """Minimal parent whose enabled_toolsets do NOT contain the composite
    'hermes-cli' name or 'kanban' — mirroring the real Kensei parent where
    platform_toolsets.cli lists individual toolset names.  This is what makes
    the composite 'hermes-cli' fail to expand and reproduce the starvation
    shape described in the incident."""
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = __import__("threading").Lock()
    parent.session_id = "parent-session"
    parent._subagent_id = None
    parent._current_task_id = "parent-task"
    parent.enabled_toolsets = list(enabled_toolsets)
    parent.valid_tool_names = [n for ts in enabled_toolsets for n in _tool_names_for(ts)]
    return parent


def _tool_names_for(ts_name):
    from toolsets import TOOLSETS

    return list(TOOLSETS.get(ts_name, {}).get("tools", []))


def _profile_cfg(toolsets):
    return {
        "name": "testprofile",
        "config": {"model": {"default": "m"}, "toolsets": toolsets, "skills": {}},
        "soul_md": None,
        "skills": [],
    }


class TestToolsetStarvationBackstop:
    """Real _build_child_agent must never resolve an empty child toolset.

    The legacy intersection against a narrower parent silently dropped
    'hermes-cli' (composite that fails to expand) leaving the child with only
    ambient tools.  The backstop must rescue that to the parent's bounded set.
    """

    def _enabled(self, MockAgent):
        # MagicMock() doesn't store constructor kwargs as attributes — the
        # resolved child toolset is the enabled_toolsets kwarg passed to AIAgent.
        return MockAgent.call_args.kwargs["enabled_toolsets"]

    def test_composite_only_profile_falls_back_to_parent_set(self, caplog):
        import logging

        parent = _mock_parent(["terminal", "file", "web", "skills", "session_search"])
        with caplog.at_level(logging.WARNING):
            with patch("run_agent.AIAgent") as MockAgent:
                MockAgent.return_value = MagicMock()
                _build_child_agent(
                    task_index=0, goal="g", context=None, toolsets=None,
                    model=None, max_iterations=10, task_count=1,
                    parent_agent=parent, profile="octacon",
                    profile_content=_profile_cfg(["hermes-cli", "kanban"]),
                )
        enabled = self._enabled(MockAgent)
        # Not empty — the backstop rescued it.
        assert enabled
        # Equals the parent's bounded set (blocked tools stripped).
        assert set(enabled) == set(_strip_blocked_tools(parent.enabled_toolsets))
        assert any("falling back" in r.message for r in caplog.records)

    def test_audit_toolset_fixes_composite_failure(self):
        """When the profile also declares 'audit', the audit toolset survives
        the intersection (its tools are a strict subset of the parent's) and
        is the only resolved toolset — proving the toolsets.py addition fixes
        the real bug without the backstop."""
        parent = _mock_parent(["terminal", "file", "web", "skills", "session_search"])
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=None,
                model=None, max_iterations=10, task_count=1,
                parent_agent=parent, profile="octacon",
                profile_content=_profile_cfg(["hermes-cli", "kanban", "audit"]),
            )
        assert self._enabled(MockAgent) == ["audit"]

    def test_unresolvable_explicit_toolsets_fall_back_not_empty(self):
        """Explicit (test-only) toolsets that don't resolve to anything must
        fall back to the parent's bounded set, never []."""
        parent = _mock_parent(["terminal", "file", "web", "skills", "session_search"])
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="g", context=None,
                toolsets=["nonexistent-a", "nonexistent-b"],
                model=None, max_iterations=10, task_count=1,
                parent_agent=parent,
            )
        assert self._enabled(MockAgent)
        assert set(self._enabled(MockAgent)) == set(
            _strip_blocked_tools(parent.enabled_toolsets)
        )

    def test_unscoped_call_unchanged(self):
        """Regression: an unscoped call (toolsets=None, no profile) behaves
        exactly as before — inherits the parent's bounded set directly."""
        parent = _mock_parent(["terminal", "file", "web", "skills", "session_search"])
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=None,
                model=None, max_iterations=10, task_count=1,
                parent_agent=parent,
            )
        assert set(self._enabled(MockAgent)) == set(
            _strip_blocked_tools(parent.enabled_toolsets)
        )

