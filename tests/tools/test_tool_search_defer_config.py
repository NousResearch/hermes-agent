"""Regression guards for the ``tools.tool_search.defer`` config key (#116404).

The key is read by ``ToolSearchConfig.from_raw`` on every assembly, so it must be
registered in ``DEFAULT_CONFIG`` (visible to ``hermes config`` / ``config get``,
coerced by the set-value guardrail, carried by migrations) and the curated
default it declares must be the one the assembly actually obeys — one source of
truth, not two lists that can drift.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _schema_defer():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    return DEFAULT_CONFIG.get("tools", {}).get("tool_search", {}).get("defer")


class TestDeferKeyRegistered:
    def test_defer_key_registered_as_list(self):
        raw = _schema_defer()
        assert isinstance(raw, list), (
            "tools.tool_search.defer must be registered in DEFAULT_CONFIG as a list — "
            "a string default would make the set-value guardrail echo user input "
            "verbatim instead of parsing list literals (#116404)")
        assert all(isinstance(n, str) and n.strip() and n == n.strip() for n in raw)

    def test_curated_default_derives_from_schema(self):
        """Single source of truth: the runtime default IS the schema default."""
        import tools.tool_search as ts
        assert ts._DEFAULT_DEFERRED_TOOLS == frozenset(
            str(n).strip() for n in _schema_defer() if str(n).strip())


class TestDeferParsing:
    def test_explicit_list_overrides_curated_default(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"defer": ["terminal", "todo_list"]})
        assert cfg.defer_tools == frozenset({"terminal", "todo_list"})

    def test_empty_list_defers_no_core_tools(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"defer": []})
        assert cfg.defer_tools == frozenset()
        assert cfg.effective_defer_tools == frozenset()

    def test_non_list_value_falls_back_to_curated_default(self):
        """A hand-edited YAML scalar (not a list) must not zero the curated set."""
        from tools.tool_search import ToolSearchConfig, _DEFAULT_DEFERRED_TOOLS
        cfg = ToolSearchConfig.from_raw({"defer": "todo_list"})
        assert cfg.defer_tools is None
        assert cfg.effective_defer_tools == _DEFAULT_DEFERRED_TOOLS

    def test_names_are_stripped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"defer": [" todo_list ", "", "session_search"]})
        assert cfg.defer_tools == frozenset({"todo_list", "session_search"})


class TestDeferEligibility:
    def test_deferred_name_wins_over_core_tool_default(self):
        """The ``defer`` set is checked BEFORE the core-tool set: naming a core
        tool there is the supported way to shrink the built-in surface."""
        from tools.tool_search import is_deferrable_tool_name
        assert is_deferrable_tool_name("todo_list", frozenset({"todo_list"}))

    def test_core_tool_not_in_defer_set_stays_direct(self):
        from tools.tool_search import is_deferrable_tool_name
        assert not is_deferrable_tool_name("terminal", frozenset({"todo_list"}))

    def test_curated_defaults_are_deferrable(self):
        from tools.tool_search import _DEFAULT_DEFERRED_TOOLS, is_deferrable_tool_name
        assert _DEFAULT_DEFERRED_TOOLS, "curated default defer set must not be empty"
        for name in sorted(_DEFAULT_DEFERRED_TOOLS):
            assert is_deferrable_tool_name(name, _DEFAULT_DEFERRED_TOOLS), name
