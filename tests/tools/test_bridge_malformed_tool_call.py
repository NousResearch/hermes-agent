"""Regression tests for malformed ``tool_call`` bridge invocations (#103752).

Qwen2.5 via Ollama aims at a real tool but emits the bridge call with the arguments
flattened at the top level — no ``name``, no ``arguments`` wrapper. Two things had to
hold for the reporter's session to stop looping forever:

1. the rejection must be *actionable* (echo the caller's own arguments in the valid
   shape), so a model that can self-correct does so on the next turn;
2. the turn must terminate even for a model that does not — the pre-existing
   ``repeated_exact_failure_block`` is gated on ``hard_stop_enabled``, which is off by
   default on the interactive platforms where this bites.

The parser half is exercised with structured payloads (no model needed). The loop
half is driven through the real conversation loop with a mocked client that re-sends
the identical malformed call every turn, which is exactly the observed behavior.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# The exact payload from the issue: the arguments the model meant for web_search,
# flattened onto the tool_call envelope with no name.
FLAT_PAYLOAD = {"query": "weather forecast New York, NY today"}


# ---------------------------------------------------------------------------
# 1. The rejection is actionable
# ---------------------------------------------------------------------------


class TestMalformedToolCallRejection:
    def test_flattened_payload_error_echoes_the_callers_own_arguments(self):
        """The error restates the valid shape with the model's OWN arguments inside,
        instead of only naming the missing field (which left it re-sending forever)."""
        from tools.tool_search_validation import normalize_tool_call_entries

        entries, err = normalize_tool_call_entries(FLAT_PAYLOAD)
        assert entries == [] and err
        assert "requires a 'name'" in err
        # The correction carries the real arguments, nested correctly.
        assert '"arguments":{"query":"weather forecast New York, NY today"}' in err
        assert '"name":"<tool name>"' in err
        # ...and it is parseable, so the model can copy it verbatim.
        start = err.index('{"name":')
        echoed = json.loads(err[start:err.index(". If the tool", start)])
        assert echoed == {"name": "<tool name>", "arguments": FLAT_PAYLOAD}

    def test_flattened_payload_error_names_the_direct_call_exit(self):
        """A tool that was never deferred should be called directly; the error says so."""
        from tools.tool_search_validation import normalize_tool_call_entries

        _, err = normalize_tool_call_entries(FLAT_PAYLOAD)
        assert "call it directly instead of via tool_call" in err

    @pytest.mark.parametrize(
        "payload,expected_prefix",
        [
            ({"calls": [{"arguments": {"query": "x"}}]}, "calls[0]"),
            ({"calls": json.dumps({"query": "x"})}, "calls[0]"),
            (FLAT_PAYLOAD, "tool_call"),
        ],
    )
    def test_every_no_name_shape_gets_the_same_correction(self, payload, expected_prefix):
        """Top-level, in-batch and stringified-envelope flattenings all get the echoed
        correction — the model cannot tell these shapes apart, so neither can we."""
        from tools.tool_search_validation import normalize_tool_call_entries

        _, err = normalize_tool_call_entries(payload)
        assert err.startswith(expected_prefix)
        assert "requires a 'name'" in err
        # The echoed correction nests the caller's own arguments, whatever shape
        # they arrived in. The flat payload carries the issue's real query.
        expected_args = '"query":' + json.dumps(
            "weather forecast New York, NY today" if payload is FLAT_PAYLOAD else "x",
            ensure_ascii=False)
        assert '"arguments":{' + expected_args + "}" in err
        assert '"name":"<tool name>"' in err

    def test_oversized_echo_collapses_to_a_placeholder(self):
        """A huge flattened payload must not bloat the error (it repeats every turn)."""
        from tools.tool_search_validation import _ECHO_ARGS_MAX_CHARS, normalize_tool_call_entries

        _, err = normalize_tool_call_entries({"blob": "x" * (_ECHO_ARGS_MAX_CHARS + 10)})
        assert '"arguments":{...}' in err
        assert len(err) < _ECHO_ARGS_MAX_CHARS + 400

    def test_well_formed_calls_still_parse(self):
        """The tolerance is on the error message only — no shape is newly accepted."""
        from tools.tool_search_validation import normalize_tool_call_entries

        assert normalize_tool_call_entries(
            {"name": "todo_list", "arguments": {"todos": []}}) == (
            [{"name": "todo_list", "arguments": {"todos": []}}], None)
        assert normalize_tool_call_entries(
            {"calls": [{"name": "todo_list", "arguments": {}}]}) == (
            [{"name": "todo_list", "arguments": {}}], None)

    def test_resolve_underlying_call_reports_the_actionable_error(self):
        """The dispatcher-facing resolver carries the same message."""
        from tools.tool_search import resolve_underlying_call

        name, args, err = resolve_underlying_call(FLAT_PAYLOAD)
        assert name is None and args == {}
        assert "requires a 'name'" in err and '"arguments":{"query":' in err


# ---------------------------------------------------------------------------
# 2. The loop terminates
# ---------------------------------------------------------------------------


def _mock_tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason)],
        model="test/model", usage=None)


def _agent_with_bridge():
    """An agent whose tool list includes the tool_search bridge, so ``tool_call`` is a
    valid model-facing name (the issue's environment) and the call reaches the bridge
    instead of being rejected earlier as an unknown tool."""
    from tests.agent.test_tool_call_guardrail_runtime import _make_agent
    import model_tools
    from tools.tool_search import _DEFAULT_DEFERRED_TOOLS

    agent = _make_agent("web_search", sorted(_DEFAULT_DEFERRED_TOOLS)[0])
    agent.tools = model_tools.get_tool_definitions(
        enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True)
    agent.valid_tool_names = {t["function"]["name"] for t in agent.tools}
    assert "tool_call" in agent.valid_tool_names
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_identical_malformed_calls(turns: int):
    """Drive the real conversation loop with a model that re-sends the identical
    malformed ``tool_call`` every turn — the observed failure mode."""
    from agent.tool_guardrails import _BRIDGE_UNRESOLVED_CAP

    agent = _agent_with_bridge()
    flat = json.dumps(FLAT_PAYLOAD)
    responses = [
        _mock_response(content="", finish_reason="tool_calls",
                       tool_calls=[_mock_tool_call("tool_call", flat, f"c{i}")])
        for i in range(turns)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop"))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What is the weather forecast for New York, NY today?")
    return agent, result


class TestUnresolvedBridgeCallLoopTerminates:
    def test_loop_is_bounded_and_halts_with_actionable_guidance(self):
        """A model that never corrects gets a bounded number of attempts, then a hard
        stop — not the 13+ identical retries the reporter observed."""
        agent, result = _run_identical_malformed_calls(turns=12)

        # Bounded: the cap (plus the one blocked call) is all the model gets.
        assert agent.client.chat.completions.create.call_count == 4
        assert result["turn_exit_reason"] == "guardrail_halt"
        assert "stopped retrying" in result["final_response"]

        tool_results = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
        # The first `cap` results are the actionable parser rejection. The tool result
        # is a JSON-encoded error object, so the message's quotes arrive escaped.
        for content in tool_results[:3]:
            assert "requires a 'name'" in content
            assert '\\"arguments\\":{\\"query\\":' in content
        # ...and the last one is the block that ends the loop.
        assert "bridge_unresolved_cap" in tool_results[-1]
        assert "never ran" in tool_results[-1]

    def test_block_fires_without_the_opt_in_hard_stop_config(self):
        """The cap is detector-independent: it holds on the default interactive
        config, where ``hard_stop_enabled`` is False and the pre-existing
        ``repeated_exact_failure_block`` never fires."""
        from agent.tool_guardrails import _BRIDGE_UNRESOLVED_CAP

        agent = _agent_with_bridge()
        assert agent._tool_guardrails.config.hard_stop_enabled is False

        flat = {"query": "x"}
        # `cap - 1` unresolved calls are still inside the budget...
        for _ in range(_BRIDGE_UNRESOLVED_CAP - 1):
            assert agent._tool_guardrails.before_call("tool_call", flat).action == "allow"
            agent._tool_guardrails.after_call(
                "tool_call", flat, json.dumps({"error": "requires a 'name'"}), failed=True)
            assert agent._tool_guardrails._bridge_unresolved_count <= _BRIDGE_UNRESOLVED_CAP - 1
        # ...the `cap`-th failure fills the budget, and the next call is blocked.
        agent._tool_guardrails.after_call(
            "tool_call", flat, json.dumps({"error": "requires a 'name'"}), failed=True)
        assert agent._tool_guardrails._bridge_unresolved_count == _BRIDGE_UNRESOLVED_CAP
        decision = agent._tool_guardrails.before_call("tool_call", flat)
        assert decision.action == "block"
        assert decision.code == "bridge_unresolved_cap"
        assert decision.count == _BRIDGE_UNRESOLVED_CAP

    def test_successful_tool_call_is_never_capped(self):
        """Only unresolved (failed) bridge calls count; a working tool_call runs freely."""
        from agent.tool_guardrails import _BRIDGE_UNRESOLVED_CAP

        agent = _agent_with_bridge()
        guardrails = agent._tool_guardrails
        good = {"name": "todo_list", "arguments": {"todos": []}}
        for _ in range(_BRIDGE_UNRESOLVED_CAP * 3):
            assert guardrails.before_call("tool_call", good).action == "allow"
            guardrails.after_call("tool_call", good, json.dumps({"ok": True}), failed=False)
        assert guardrails._bridge_unresolved_count == 0
        assert guardrails.before_call("tool_call", good).action == "allow"

    def test_counter_resets_between_turns(self):
        """The cap is per turn: a fresh turn gets its full budget of attempts."""
        from agent.tool_guardrails import _BRIDGE_UNRESOLVED_CAP

        agent = _agent_with_bridge()
        guardrails = agent._tool_guardrails
        flat = {"query": "x"}
        # The real parser rejection, so the counter sees the shape it must recognize.
        from tools.tool_search_validation import normalize_tool_call_entries
        _, err = normalize_tool_call_entries(flat)
        rejection = json.dumps({"error": err})
        for _ in range(_BRIDGE_UNRESOLVED_CAP + 2):
            guardrails.after_call("tool_call", flat, rejection, failed=True)
        assert guardrails.before_call("tool_call", flat).action == "block"

        guardrails.reset_for_turn()
        assert guardrails._bridge_unresolved_count == 0
        assert guardrails.before_call("tool_call", flat).action == "allow"

    def test_a_failed_tool_call_that_did_resolve_is_not_capped(self):
        """A well-formed bridge call whose target tool then failed reached a tool, so
        it must not feed the unresolved cap — otherwise a run of real failures inside
        one turn would be blocked as if the bridge itself were broken."""
        from agent.tool_guardrails import _BRIDGE_UNRESOLVED_CAP

        agent = _agent_with_bridge()
        guardrails = agent._tool_guardrails
        good = {"name": "todo_list", "arguments": {"todos": []}}
        # A resolved call that failed: the error names the tool, not the missing name.
        resolved_failure = json.dumps({"error": "todo_list failed: unknown todo list id"})
        for _ in range(_BRIDGE_UNRESOLVED_CAP * 2):
            assert guardrails.before_call("tool_call", good).action == "allow"
            guardrails.after_call("tool_call", good, resolved_failure, failed=True)
        assert guardrails._bridge_unresolved_count == 0
        assert guardrails.before_call("tool_call", good).action == "allow"
