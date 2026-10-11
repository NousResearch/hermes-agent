"""A core tool stripped from this session's toolset must not be told to 'call it directly'.

`_HERMES_CORE_TOOLS` is static, so `not_deferrable_error` matched any core tool by name —
including `clarify` inside a subagent, where it is toolset-blocked and no direct call can
ever succeed. The static list cannot distinguish "callable, wrong door" from "not here at all",
so the model was sent into a retry loop on a confident false instruction.

Regression test for that: the fix is only proven if it goes RED when reverted.
"""

from tools.tool_search_validation import not_deferrable_error


class TestNotDeferrableSessionScope:
    def test_core_tool_absent_from_session_reports_unavailable(self):
        """The subagent case: clarify is core but blocked here."""
        msg = not_deferrable_error("clarify", session_tool_names={"read_file", "terminal"})
        assert "not available in this execution context" in msg
        assert "directly-listed" not in msg, (
            "must not tell the model to call a tool this session cannot reach"
        )
        assert "retry" in msg.lower(), "should tell the model not to retry"

    def test_core_tool_present_in_session_keeps_door_guidance(self):
        """Top-level case unchanged: clarify IS callable directly here."""
        msg = not_deferrable_error("clarify", session_tool_names={"clarify", "read_file"})
        assert "directly-listed" in msg
        assert "Call it directly" in msg

    def test_no_session_context_preserves_original_behaviour(self):
        """Callers with no session context (display layer, recorder) are unaffected."""
        msg = not_deferrable_error("clarify")
        assert "directly-listed" in msg

    def test_unknown_tool_still_gets_discovery_hint(self):
        """The unrelated branch must not regress: bare-suffix MCP names."""
        msg = not_deferrable_error("definitely_not_a_real_tool", session_tool_names={"clarify"})
        assert "not a known tool name" in msg

    def test_session_names_accepts_any_iterable(self):
        """Generators must work, not just sets — the dispatcher passes a frozenset but
        the annotation promises Iterable and a generator is the obvious caller mistake."""
        msg = not_deferrable_error("clarify", session_tool_names=(n for n in ["read_file"]))
        assert "not available in this execution context" in msg
