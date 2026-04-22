"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from tools.delegate_tool import _strip_blocked_tools, _expand_parent_enabled_toolsets, _emit_parent_console


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

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "web"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["terminal", "web"]

    def test_no_toolsets_requested_inherits_parent(self):
        """When toolsets is None/empty, child inherits parent's set."""
        parent_toolsets = ["terminal", "file", "web"]
        child = _strip_blocked_tools(parent_toolsets)
        assert "terminal" in child
        assert "file" in child
        assert "web" in child

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


class TestExpandParentEnabledToolsets:
    """Preset names (e.g. ``hermes-acp``) must expand to individual toolsets
    before subagent intersection. Without this, an ACP-hosted parent would
    always pass an empty toolset to children — a regression that stripped
    every tool from subagents spawned via ``delegate_task``."""

    def test_preset_expands_to_individual_toolsets(self):
        """Preset bundle expands to its underlying individual toolsets."""
        expanded = _expand_parent_enabled_toolsets(["hermes-acp"])
        # hermes-acp bundles web, browser, terminal, file, skills, vision,
        # session_search, todo, memory, code_execution, delegation — exact
        # membership depends on toolsets.py, so only assert the key ones
        # the delegate-scoping test cares about.
        assert "web" in expanded
        assert "browser" in expanded
        assert "file" in expanded
        assert "terminal" in expanded

    def test_individual_toolset_is_idempotent(self):
        """Passing an already-individual toolset name returns itself."""
        expanded = _expand_parent_enabled_toolsets(["web", "terminal"])
        assert expanded == {"web", "terminal"}

    def test_mixed_preset_and_individual_toolsets(self):
        """Mix of preset + individual names all expand correctly."""
        expanded = _expand_parent_enabled_toolsets(["hermes-acp", "rl"])
        assert "browser" in expanded  # from hermes-acp
        assert "rl" in expanded       # individual passthrough

    def test_unknown_name_kept_literal(self):
        """Unknown/legacy names are preserved so downstream code sees them."""
        expanded = _expand_parent_enabled_toolsets(["definitely-not-a-real-toolset"])
        assert expanded == {"definitely-not-a-real-toolset"}

    def test_preset_parent_intersects_with_llm_requested(self):
        """End-to-end: preset parent + LLM-requested individual toolsets
        must produce the correct intersection. Regression for the bug where
        ``parent_toolsets = {"hermes-acp"}`` trivially rejected every request.
        """
        parent_toolsets = _expand_parent_enabled_toolsets(["hermes-acp"])
        requested = ["browser", "terminal", "web"]
        scoped = _strip_blocked_tools([t for t in requested if t in parent_toolsets])
        assert sorted(scoped) == ["browser", "terminal", "web"]


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

    def test_falls_back_to_stdout_when_no_safe_print(self, capsys):
        parent = SimpleNamespace()
        _emit_parent_console(parent, "  ✓ [1/3] fallback path")
        captured = capsys.readouterr()
        assert "fallback path" in captured.out

    def test_falls_back_to_stdout_when_safe_print_raises(self, capsys):
        def raiser(_line):
            raise RuntimeError("boom")

        parent = SimpleNamespace(_safe_print=raiser)
        _emit_parent_console(parent, "  ✓ [2/3] fallback on exception")
        captured = capsys.readouterr()
        assert "fallback on exception" in captured.out

    def test_non_callable_safe_print_is_ignored(self, capsys):
        """Defensive: if _safe_print is set but not callable, fall back."""
        parent = SimpleNamespace(_safe_print="not-a-function")
        _emit_parent_console(parent, "  ✓ [3/3] non-callable guard")
        captured = capsys.readouterr()
        assert "non-callable guard" in captured.out
