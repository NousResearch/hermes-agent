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


class TestReadOnlyDelegation:
    def test_build_child_agent_read_only_strips_side_effect_toolsets(self, monkeypatch):
        import sys
        import types
        import tools.delegate_tool as dt

        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.session_id = "child-session"
                self.enabled_toolsets = kwargs.get("enabled_toolsets")

        fake_mod = types.ModuleType("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)
        monkeypatch.setattr(dt, "_load_config", lambda: {})
        monkeypatch.setattr(dt, "_build_child_progress_callback", lambda *a, **k: None)

        parent = SimpleNamespace(
            enabled_toolsets=[
                "terminal",
                "file",
                "web",
                "browser",
                "skills",
                "memory",
                "delegation",
                "cronjob",
                "code_execution",
                "computer_use",
            ],
            model="parent-model",
            provider="test",
            base_url="http://test.local",
            api_key="key",
            _delegate_depth=0,
            session_id="parent-session",
        )

        child = dt._build_child_agent(
            task_index=0,
            goal="read the repo",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=1,
            task_count=1,
            parent_agent=parent,
            capability="read_only",
        )

        assert set(child.enabled_toolsets) == {
            "browser_read",
            "file_read",
            "skills_read",
            "web",
        }
        assert captured["enabled_toolsets"] == child.enabled_toolsets
        assert "terminal" not in child.enabled_toolsets
        assert "file" not in child.enabled_toolsets
        assert "skills" not in child.enabled_toolsets
        assert "memory" not in child.enabled_toolsets
        assert "delegation" not in child.enabled_toolsets
        assert "cronjob" not in child.enabled_toolsets
        assert "code_execution" not in child.enabled_toolsets
        assert "computer_use" not in child.enabled_toolsets

    def test_dispatch_rejects_tool_not_available_in_enabled_toolsets(self, tmp_path):
        import json
        from model_tools import handle_function_call

        target = tmp_path / "should-not-exist.txt"
        raw = handle_function_call(
            "write_file",
            {"path": str(target), "content": "nope"},
            task_id="read-only-adversarial",
            enabled_toolsets=["file_read"],
            skip_pre_tool_call_hook=True,
        )

        result = json.loads(raw)
        assert "not available in this session" in result["error"]
        assert not target.exists()


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
