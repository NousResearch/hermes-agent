"""Unit tests for ``tools.swarm_tool`` — the native Hermes swarm spawner.

These tests mock ``delegate_task`` so we exercise swarm_tool's logic
(validation, topology dispatch, prelude composition, result wrapping)
without spinning up real subagent processes.
"""
from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.swarm_tool import (
    MAX_AGENTS_PER_SWARM,
    SWARM_RUN_SCHEMA,
    VALID_TOPOLOGIES,
    _build_swarm_prelude,
    _peer_summaries,
    _validate_agents,
    _validate_topology,
    _wrap_delegate_result,
    check_swarm_run_requirements,
    swarm_run,
)


# ── Test helpers ──────────────────────────────────────────────────────────


def _mock_parent():
    """Mock parent with the attrs delegate_task touches."""
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _fake_delegate_response(*summaries: str) -> str:
    """Build a JSON string mirroring delegate_task's return shape."""
    return json.dumps({
        "results": [
            {"summary": s, "ok": True}
            for s in summaries
        ],
    })


# ── Schema / requirements ─────────────────────────────────────────────────


class TestSchema(unittest.TestCase):
    def test_check_always_true(self):
        self.assertTrue(check_swarm_run_requirements())

    def test_schema_shape(self):
        self.assertEqual(SWARM_RUN_SCHEMA["name"], "swarm_run")
        props = SWARM_RUN_SCHEMA["parameters"]["properties"]
        self.assertIn("agents", props)
        self.assertIn("topology", props)
        self.assertIn("title", props)
        self.assertIn("shared_context", props)
        self.assertEqual(SWARM_RUN_SCHEMA["parameters"]["required"], ["agents"])
        self.assertEqual(props["topology"]["enum"], list(VALID_TOPOLOGIES))


# ── Validation ────────────────────────────────────────────────────────────


class TestValidateAgents(unittest.TestCase):
    def test_rejects_non_list(self):
        with self.assertRaises(ValueError):
            _validate_agents("not a list")
        with self.assertRaises(ValueError):
            _validate_agents(None)

    def test_rejects_empty_list(self):
        with self.assertRaises(ValueError):
            _validate_agents([])

    def test_rejects_missing_type(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_agents([{"goal": "do thing"}])
        self.assertIn("type", str(ctx.exception))

    def test_rejects_missing_goal(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_agents([{"type": "researcher"}])
        self.assertIn("goal", str(ctx.exception))

    def test_accepts_agent_type_alias(self):
        """``agent_type`` is accepted as an alias for ``type`` for the LLMs
        that map directly from delegate_task's vocabulary."""
        out = _validate_agents([{"agent_type": "researcher", "goal": "g"}])
        self.assertEqual(out[0]["type"], "researcher")

    def test_passes_through_optional_fields(self):
        out = _validate_agents([{
            "type": "coder",
            "goal": "fix bug",
            "context": "extra",
            "model": "claude-haiku-4-5",
            "toolsets": ["terminal"],
            "agent_id": "custom-id",
        }])
        self.assertEqual(out[0]["context"], "extra")
        self.assertEqual(out[0]["model"], "claude-haiku-4-5")
        self.assertEqual(out[0]["toolsets"], ["terminal"])
        self.assertEqual(out[0]["agent_id"], "custom-id")

    def test_rejects_non_dict_entries(self):
        with self.assertRaises(ValueError):
            _validate_agents(["just a string"])

    def test_rejects_swarm_too_large(self):
        big = [
            {"type": "researcher", "goal": f"task {i}"}
            for i in range(MAX_AGENTS_PER_SWARM + 1)
        ]
        with self.assertRaises(ValueError) as ctx:
            _validate_agents(big)
        self.assertIn("too many", str(ctx.exception).lower())


class TestValidateTopology(unittest.TestCase):
    def test_default_is_parallel(self):
        self.assertEqual(_validate_topology(None), "parallel")
        self.assertEqual(_validate_topology(""), "parallel")

    def test_known_values_pass(self):
        for t in VALID_TOPOLOGIES:
            self.assertEqual(_validate_topology(t), t)

    def test_case_insensitive(self):
        self.assertEqual(_validate_topology("HIERARCHICAL"), "hierarchical")

    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):
            _validate_topology("quantum")


# ── Prelude composition ──────────────────────────────────────────────────


class TestPrelude(unittest.TestCase):
    def _peers(self):
        return [
            {"agent_id": "a1-researcher", "agent_type": "researcher",
             "goal": "find docs"},
            {"agent_id": "a2-reviewer", "agent_type": "reviewer",
             "goal": "review docs"},
        ]

    def test_includes_identity(self):
        text = _build_swarm_prelude(
            swarm_id="sw-x", agent_id="a1-researcher",
            agent_type="researcher", topology="parallel",
            peers=self._peers(), role_in_swarm="worker",
        )
        self.assertIn("a1-researcher", text)
        self.assertIn("researcher", text)
        self.assertIn("sw-x", text)
        self.assertIn("parallel", text)

    def test_includes_peer_list(self):
        text = _build_swarm_prelude(
            swarm_id="sw-x", agent_id="a1-researcher",
            agent_type="researcher", topology="parallel",
            peers=self._peers(), role_in_swarm="worker",
        )
        self.assertIn("a2-reviewer", text)
        self.assertIn("reviewer", text)

    def test_mentions_swarm_mcp_tools(self):
        """Children must be told the swarm tools exist — tests catch
        regressions where the prelude drops the coordination contract.

        Names carry a doubled ``swarm_`` because the MCP server is named
        ``hermes-swarm`` and the tool inside is e.g. ``swarm_memory_store``.
        Emitting the singular form would mislead children into hitting the
        auto-repair fallback on every call.
        """
        text = _build_swarm_prelude(
            swarm_id="sw-x", agent_id="a1", agent_type="t",
            topology="parallel", peers=[], role_in_swarm="worker",
        )
        self.assertIn("mcp_hermes_swarm_swarm_memory_store", text)
        self.assertIn("mcp_hermes_swarm_swarm_broadcast", text)
        self.assertIn("mcp_hermes_swarm_swarm_inbox", text)
        self.assertIn("mcp_hermes_swarm_swarm_update_agent", text)
        # Guard against regression to the singular form.  A standalone
        # `mcp_hermes_swarm_memory_store` (no double swarm_) is the wrong
        # name — fail if it shows up.
        self.assertNotIn("mcp_hermes_swarm_memory_store(", text)


# ── Result wrapping ───────────────────────────────────────────────────────


class TestWrapDelegateResult(unittest.TestCase):
    def test_wraps_results_with_agent_metadata(self):
        agents = [
            {"agent_id": "a1", "type": "researcher", "goal": "x"},
            {"agent_id": "a2", "type": "reviewer", "goal": "y"},
        ]
        raw = _fake_delegate_response("found 5 things", "looks good")
        out = _wrap_delegate_result(raw, agents)
        self.assertEqual(len(out["results"]), 2)
        self.assertEqual(out["results"][0]["agent_id"], "a1")
        self.assertEqual(out["results"][0]["agent_type"], "researcher")
        self.assertEqual(out["results"][0]["summary"], "found 5 things")
        self.assertTrue(out["results"][0]["ok"])

    def test_handles_error_response(self):
        raw = json.dumps({"error": "delegation paused"})
        out = _wrap_delegate_result(raw, [])
        self.assertEqual(out["results"], [])
        self.assertIn("delegation paused", out["error"])

    def test_handles_non_json(self):
        out = _wrap_delegate_result("not json at all", [])
        self.assertEqual(out["results"], [])
        self.assertIn("non-JSON", out["error"])

    def test_carries_through_cost_metadata(self):
        agents = [{"agent_id": "a1", "type": "r", "goal": "x"}]
        raw = json.dumps({
            "results": [{
                "summary": "done",
                "ok": True,
                "model": "claude-haiku-4-5",
                "duration_s": 12.3,
                "cost_usd": 0.04,
                "input_tokens": 1200,
                "output_tokens": 300,
            }],
        })
        out = _wrap_delegate_result(raw, agents)
        r0 = out["results"][0]
        self.assertEqual(r0["model"], "claude-haiku-4-5")
        self.assertEqual(r0["duration_s"], 12.3)
        self.assertEqual(r0["cost_usd"], 0.04)
        self.assertEqual(r0["input_tokens"], 1200)


class TestPeerSummaries(unittest.TestCase):
    def test_extracts_peer_visible_fields_only(self):
        agents = [
            {"agent_id": "a1", "type": "researcher", "goal": "find",
             "context": "secret", "model": "haiku"},
            {"agent_id": "a2", "type": "coder", "goal": "build",
             "context": "secret", "model": "sonnet"},
        ]
        peers = _peer_summaries(agents)
        # Peers see id, type, goal — not context or model (those are
        # per-agent private routing concerns).
        self.assertEqual(set(peers[0].keys()), {"agent_id", "agent_type", "goal"})


# ── End-to-end: swarm_run dispatch ────────────────────────────────────────


class TestSwarmRunRequiresParent(unittest.TestCase):
    def test_no_parent_returns_error(self):
        out = json.loads(swarm_run(
            agents=[{"type": "researcher", "goal": "x"}],
            parent_agent=None,
        ))
        self.assertIn("error", out)


class TestSwarmRunValidationErrors(unittest.TestCase):
    def test_missing_agents_errors(self):
        parent = _mock_parent()
        out = json.loads(swarm_run(parent_agent=parent))
        self.assertIn("error", out)

    def test_unknown_topology_errors(self):
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[{"type": "r", "goal": "x"}],
            topology="bogus",
            parent_agent=parent,
        ))
        self.assertIn("error", out)


class TestSwarmRunDispatch(unittest.TestCase):
    @patch("tools.delegate_tool.delegate_task")
    def test_parallel_calls_delegate_task_once_with_full_batch(self, mock_dt):
        mock_dt.return_value = _fake_delegate_response("r1", "r2", "r3")
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[
                {"type": "researcher", "goal": "g1"},
                {"type": "code-analyzer", "goal": "g2"},
                {"type": "code-analyzer", "goal": "g3"},
            ],
            topology="parallel",
            parent_agent=parent,
        ))
        # Single batched call with 3 tasks.
        self.assertEqual(mock_dt.call_count, 1)
        kwargs = mock_dt.call_args.kwargs
        self.assertEqual(len(kwargs["tasks"]), 3)
        # Each task got the swarm prelude in its context.
        for t in kwargs["tasks"]:
            self.assertIn("SWARM COORDINATION CONTEXT", t["context"])
        # All 3 results surface up.
        self.assertEqual(len(out["results"]), 3)
        self.assertEqual(out["topology"], "parallel")
        self.assertTrue(out["swarm_id"].startswith("sw-"))

    @patch("tools.delegate_tool.delegate_task")
    def test_sequential_calls_delegate_task_per_agent(self, mock_dt):
        mock_dt.side_effect = [
            _fake_delegate_response("first output"),
            _fake_delegate_response("second output, saw first"),
        ]
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[
                {"type": "researcher", "goal": "find"},
                {"type": "reviewer", "goal": "review"},
            ],
            topology="sequential",
            parent_agent=parent,
        ))
        self.assertEqual(mock_dt.call_count, 2)
        # Second call's context must include the first agent's output.
        second_call_kwargs = mock_dt.call_args_list[1].kwargs
        second_context = second_call_kwargs["tasks"][0]["context"]
        self.assertIn("first output", second_context)
        self.assertIn("Prior agent", second_context)
        # Both results in output.
        self.assertEqual(len(out["results"]), 2)

    @patch("tools.delegate_tool.delegate_task")
    def test_pipeline_uses_input_framing(self, mock_dt):
        mock_dt.side_effect = [
            _fake_delegate_response("first stage output"),
            _fake_delegate_response("transformed"),
        ]
        parent = _mock_parent()
        swarm_run(
            agents=[
                {"type": "researcher", "goal": "research"},
                {"type": "coder", "goal": "implement"},
            ],
            topology="pipeline",
            parent_agent=parent,
        )
        second_context = mock_dt.call_args_list[1].kwargs["tasks"][0]["context"]
        # Pipeline framing uses the YOUR INPUT block for the most recent prior.
        self.assertIn("YOUR INPUT", second_context)
        self.assertIn("first stage output", second_context)

    @patch("tools.delegate_tool.delegate_task")
    def test_hierarchical_workers_then_synthesizer(self, mock_dt):
        mock_dt.side_effect = [
            _fake_delegate_response("worker A output", "worker B output"),
            _fake_delegate_response("synthesis"),
        ]
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[
                {"type": "code-analyzer", "goal": "analyze A"},
                {"type": "code-analyzer", "goal": "analyze B"},
                {"type": "reviewer", "goal": "summarise"},
            ],
            topology="hierarchical",
            parent_agent=parent,
        ))
        # Two delegate calls: one batched (workers), one solo (synthesizer).
        self.assertEqual(mock_dt.call_count, 2)
        self.assertEqual(len(mock_dt.call_args_list[0].kwargs["tasks"]), 2)
        self.assertEqual(len(mock_dt.call_args_list[1].kwargs["tasks"]), 1)
        # Synthesizer's context contains both workers' output blocks.
        synth_context = mock_dt.call_args_list[1].kwargs["tasks"][0]["context"]
        self.assertIn("worker A output", synth_context)
        self.assertIn("worker B output", synth_context)
        self.assertIn("WORKER OUTPUTS", synth_context)
        # All three results are returned to caller.
        self.assertEqual(len(out["results"]), 3)


class TestSwarmRunSharedContext(unittest.TestCase):
    @patch("tools.delegate_tool.delegate_task")
    def test_shared_context_reaches_every_agent(self, mock_dt):
        mock_dt.return_value = _fake_delegate_response("done", "done")
        parent = _mock_parent()
        swarm_run(
            agents=[
                {"type": "researcher", "goal": "g1"},
                {"type": "reviewer", "goal": "g2"},
            ],
            shared_context="Customer is Acme. Case 00264067.",
            parent_agent=parent,
        )
        for t in mock_dt.call_args.kwargs["tasks"]:
            self.assertIn("Acme", t["context"])
            self.assertIn("00264067", t["context"])


class TestSwarmRunIdAssignment(unittest.TestCase):
    @patch("tools.delegate_tool.delegate_task")
    def test_user_swarm_id_honored(self, mock_dt):
        mock_dt.return_value = _fake_delegate_response("ok")
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[{"type": "researcher", "goal": "x"}],
            swarm_id="case-00264067",
            parent_agent=parent,
        ))
        self.assertEqual(out["swarm_id"], "case-00264067")

    @patch("tools.delegate_tool.delegate_task")
    def test_user_agent_id_honored(self, mock_dt):
        mock_dt.return_value = _fake_delegate_response("ok")
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[{"type": "researcher", "goal": "x", "agent_id": "main-r"}],
            parent_agent=parent,
        ))
        self.assertEqual(out["results"][0]["agent_id"], "main-r")

    @patch("tools.delegate_tool.delegate_task")
    def test_auto_generated_ids_distinct_and_typed(self, mock_dt):
        mock_dt.return_value = _fake_delegate_response("a", "b", "c")
        parent = _mock_parent()
        out = json.loads(swarm_run(
            agents=[
                {"type": "researcher", "goal": "1"},
                {"type": "researcher", "goal": "2"},
                {"type": "code-analyzer", "goal": "3"},
            ],
            parent_agent=parent,
        ))
        ids = [r["agent_id"] for r in out["results"]]
        self.assertEqual(len(set(ids)), 3)  # all distinct
        # Auto-generated ids include a hint of the agent type for readability.
        self.assertIn("researcher", ids[0])
        self.assertIn("code-analyzer", ids[2])


if __name__ == "__main__":
    unittest.main()
