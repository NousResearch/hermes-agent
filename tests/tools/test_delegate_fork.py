"""delegate_task context_mode="fork": the child starts from the parent's conversation."""

import json
import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent, delegate_task
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


class TestDelegateFork(unittest.TestCase):
    def test_fork_snapshot_keeps_the_turn_and_closes_the_in_flight_delegation(self):
        """A fork copies the parent's live turn (request + completed tool rounds), answers the in-flight
        delegate_task call so the parent's request is not merged into the child's goal, and is frozen at launch."""
        from agent.agent_runtime_helpers import repair_message_sequence
        from tools.delegate_tool_fork import frame_forked_goal

        parent = _make_mock_parent()
        read_call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        fork_call = {"id": "c2", "type": "function", "function": {"name": "delegate_task", "arguments": "{}"}}
        parent._session_messages = [
            {"role": "user", "content": "Read notes.txt, then fork a worker on it."},
            {"role": "assistant", "content": "", "tool_calls": [read_call]},
            {"role": "tool", "tool_call_id": "c1", "content": "codeword AZURE"},
            {"role": "assistant", "content": "", "tool_calls": [fork_call]},
        ]
        child = MagicMock()
        with patch("run_agent.AIAgent", return_value=child):
            _build_child_agent(
                task_index=0, goal="Report the codeword", context=None, toolsets=None, model="test-model",
                max_iterations=5, parent_agent=parent, task_count=1, context_mode="fork",
            )
        parent._session_messages.append({"role": "tool", "tool_call_id": "c2", "content": "later parent result"})

        history = child._delegate_fork_history
        self.assertEqual(history[:4], parent._session_messages[:4])
        self.assertEqual((history[4]["role"], history[4]["tool_call_id"]), ("tool", "c2"))
        self.assertIn("Forked", history[4]["content"])
        self.assertEqual(len(history), 5)
        child_turn = history + [{"role": "user", "content": frame_forked_goal("Report the codeword")}]
        self.assertEqual(repair_message_sequence(None, child_turn), 0)

        # The parent's calls in the inherited prefix are not reported as the child's work.
        child._delegate_fork_history = history
        own_call = {"id": "k1", "type": "function", "function": {"name": "search_files", "arguments": "{}"}}
        result = {"final_response": "AZURE", "completed": True, "messages": child_turn + [
            {"role": "assistant", "content": "", "tool_calls": [own_call]},
            {"role": "tool", "tool_call_id": "k1", "content": "[]"},
            {"role": "assistant", "content": "AZURE"},
        ]}
        entry = _build_result_entry(child, result, 0, 1.0, _SchemaOutcome(None, None, [], 0))
        self.assertEqual([t["tool"] for t in entry["tool_trace"]], ["search_files"])

    def test_context_mode_preserves_tool_error_for_malformed_tasks(self):
        parent = _make_mock_parent(depth=0)

        for malformed in (123, "hello", None):
            with self.subTest(malformed=malformed):
                result = json.loads(
                    delegate_task(
                        tasks=[malformed],  # type: ignore[list-item]
                        context_mode="fork", parent_agent=parent
                    )
                )
                self.assertIn("error", result)
                self.assertIn("Task 0 must be an object", result["error"])
