#!/usr/bin/env python3
"""Tests for the delegation logical-outcome envelope (false-green fix).

These lock in the separation between a subagent's *lifecycle* status (the
child loop ended and produced text) and its *logical* task outcome. A
non-empty summary must NEVER be classified as verified success: the strongest
outcome this slice can assign is ``unverified``, which explicitly requires
parent verification.

The first group exercises the real ``_run_single_child`` shaping path with a
lightweight fake child (not a pure helper). The second group covers the async
re-injection formatter's neutralised rendering.

Run:  scripts/run_tests.sh tests/tools/test_delegation_evidence_gate.py
"""

import threading
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from tools.async_delegation import _push_completion_event
from tools.delegate_tool import _run_single_child, delegate_task
from tools.process_registry import _derive_result_outcome, _format_async_delegation


def _make_mock_parent():
    """Minimal parent with the attributes _run_single_child touches."""
    parent = type("P", (), {})()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._current_task_id = None
    parent._touch_activity = lambda *a, **k: None
    return parent


class FakeChild:
    """Lightweight child double whose run_conversation returns a canned loop
    result — enough for _run_single_child's post-run shaping to run for real."""

    def __init__(self, loop_result, model="fake-model"):
        self._loop_result = loop_result
        self.model = model
        self.tool_progress_callback: Any = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self._delegate_role = "leaf"

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        return self._loop_result

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "max_iterations": 50}

    def close(self):
        pass


def _shape(loop_result):
    """Run the real _run_single_child shaping path over a fake child."""
    child = FakeChild(loop_result)
    return _run_single_child(0, "do the thing", child, _make_mock_parent())


class TestRunSingleChildOutcome(unittest.TestCase):
    def test_completed_true_with_summary_is_unverified_never_success(self):
        entry = _shape({
            "final_response": "I finished the task.",
            "completed": True,
            "interrupted": False,
            "api_calls": 2,
            "messages": [],
        })
        # Lifecycle may report completed for backward compatibility...
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["exit_reason"], "completed")
        # ...but the LOGICAL outcome is only unverified — never a success class.
        self.assertEqual(entry["outcome"], "unverified")
        self.assertNotIn(entry["outcome"], ("success", "verified", "complete", "completed"))
        self.assertFalse(entry["interrupted"])

    def test_completed_false_with_summary_is_partial_max_iterations(self):
        entry = _shape({
            "final_response": "Got partway before running out of steps.",
            "completed": False,
            "interrupted": False,
            "api_calls": 50,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "partial")
        self.assertEqual(entry["exit_reason"], "max_iterations")

    def test_partial_stream_recovery_remains_partial_even_when_completed(self):
        entry = _shape({
            "final_response": "Recovered partial streamed output.",
            "completed": True,
            "failed": False,
            "turn_exit_reason": "partial_stream_recovery",
            "interrupted": False,
            "api_calls": 2,
            "messages": [],
        })
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["outcome"], "partial")
        self.assertEqual(entry["exit_reason"], "partial_stream_recovery")

    def test_explicit_child_failure_overrides_nonempty_summary(self):
        entry = _shape({
            "final_response": "Error: account credits exhausted.",
            "completed": False,
            "failed": True,
            "error": "account credits exhausted",
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        })
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["outcome"], "failed")
        self.assertEqual(entry["exit_reason"], "error")
        self.assertEqual(entry["error"], "account credits exhausted")

    def test_explicit_error_field_fails_closed_without_failed_flag(self):
        entry = _shape({
            "final_response": "Codex app-server crashed after partial work.",
            "completed": False,
            "failed": False,
            "partial": True,
            "error": "Codex app-server crashed",
            "interrupted": False,
            "api_calls": 2,
            "messages": [],
        })
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["outcome"], "failed")
        self.assertEqual(entry["exit_reason"], "error")
        self.assertEqual(entry["error"], "Codex app-server crashed")

    def test_failed_child_preserves_runtime_exit_reason_and_summary_diagnostic(self):
        entry = _shape({
            "final_response": "Ollama context window exceeded.",
            "completed": False,
            "failed": True,
            "turn_exit_reason": "context_window_exceeded",
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "failed")
        self.assertEqual(entry["exit_reason"], "context_window_exceeded")
        self.assertEqual(entry["error"], "Ollama context window exceeded.")

    def test_fatal_turn_exit_reason_fails_closed_without_failed_flag(self):
        for reason in ("empty_response_exhausted", "all_retries_exhausted_no_response"):
            with self.subTest(reason=reason):
                entry = _shape({
                    "final_response": f"Runtime failure: {reason}",
                    "completed": False,
                    "failed": False,
                    "turn_exit_reason": reason,
                    "interrupted": False,
                    "api_calls": 3,
                    "messages": [],
                })
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["outcome"], "failed")
                self.assertEqual(entry["exit_reason"], reason)
                self.assertEqual(entry["error"], f"Runtime failure: {reason}")

    def test_code_skew_exit_reasons_fail_closed_without_failed_flag(self):
        for reason in (
            "code_skew_detected",
            "code_skew_attribute_error(AIAgent missing updated member)",
        ):
            with self.subTest(reason=reason):
                entry = _shape({
                    "final_response": "Restart required after runtime code skew.",
                    "completed": False,
                    "failed": False,
                    "turn_exit_reason": reason,
                    "interrupted": False,
                    "api_calls": 1,
                    "messages": [],
                })
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["outcome"], "failed")
                self.assertEqual(entry["exit_reason"], reason)
                self.assertEqual(
                    entry["error"],
                    "Restart required after runtime code skew.",
                )

    def test_interrupted_with_summary_is_partial(self):
        entry = _shape({
            "final_response": "Partial progress captured.",
            "completed": False,
            "interrupted": True,
            "api_calls": 5,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "partial")
        self.assertTrue(entry["interrupted"])
        self.assertEqual(entry["exit_reason"], "interrupted")

    def test_interrupted_without_summary_is_failed(self):
        entry = _shape({
            "final_response": "",
            "completed": False,
            "interrupted": True,
            "api_calls": 1,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "failed")

    def test_empty_sentinel_is_failed(self):
        entry = _shape({
            "final_response": "(empty)",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "failed")

    def test_blank_summary_is_failed(self):
        entry = _shape({
            "final_response": "   ",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        })
        self.assertEqual(entry["outcome"], "failed")

    def test_tool_error_count_is_deterministic_from_messages(self):
        entry = _shape({
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 2,
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "a", "function": {"name": "terminal", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "Error: command not found"},
                {"role": "assistant", "tool_calls": [
                    {"id": "b", "function": {"name": "terminal", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "b", "content": "ok, listing complete"},
            ],
        })
        # Exactly one of the two tool results is an error — deterministic count.
        self.assertEqual(entry["tool_error_count"], 1)
        self.assertEqual(entry["exit_reason"], "completed")

    def test_progress_completion_event_carries_logical_outcome_evidence(self):
        events = []
        child = FakeChild({
            "final_response": "summary needs parent verification",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        })
        child.tool_progress_callback = (
            lambda event, *args, **kwargs: events.append((event, kwargs))
        )

        _run_single_child(0, "do the thing", child, _make_mock_parent())

        completion = [payload for event, payload in events if event == "subagent.complete"]
        self.assertEqual(len(completion), 1)
        self.assertEqual(completion[0]["outcome"], "unverified")
        self.assertEqual(completion[0]["tool_error_count"], 0)

    def test_exception_completion_event_carries_failed_envelope(self):
        class RaisingChild(FakeChild):
            def run_conversation(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        events = []
        child = RaisingChild({})
        child.tool_progress_callback = (
            lambda event, *args, **kwargs: events.append((event, kwargs))
        )

        entry = _run_single_child(0, "do the thing", child, _make_mock_parent())

        completion = [payload for event, payload in events if event == "subagent.complete"]
        self.assertEqual(entry["outcome"], "failed")
        self.assertEqual(completion[0]["outcome"], "failed")
        self.assertEqual(completion[0]["exit_reason"], "error")
        self.assertFalse(completion[0]["interrupted"])
        self.assertEqual(completion[0]["tool_error_count"], 0)


class TestBatchProgressNeutralisation(unittest.TestCase):
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._run_single_child")
    def test_batch_progress_line_never_green_checks_unverified_output(
        self, mock_run, mock_build
    ):
        mock_build.side_effect = [MagicMock(), MagicMock()]
        mock_run.side_effect = [
            {
                "task_index": 0,
                "status": "completed",
                "outcome": "unverified",
                "summary": "first self-report",
                "api_calls": 1,
                "duration_seconds": 0.1,
            },
            {
                "task_index": 1,
                "status": "completed",
                "outcome": "partial",
                "summary": "second partial report",
                "api_calls": 1,
                "duration_seconds": 0.1,
            },
        ]
        parent = MagicMock()
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "***"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "test-model"
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._memory_manager = None
        parent._delegate_spinner = MagicMock()

        delegate_task(
            tasks=[{"goal": "first"}, {"goal": "second"}],
            parent_agent=parent,
        )

        rendered = "\n".join(
            str(call.args[0])
            for call in parent._delegate_spinner.print_above.call_args_list
        )
        self.assertNotIn("✓", rendered)
        self.assertIn("⚠", rendered)
        self.assertIn("◐", rendered)


class TestAsyncEnvelopePropagation(unittest.TestCase):
    @patch("tools.async_delegation._persist_completion")
    @patch("tools.process_registry.process_registry")
    def test_single_async_event_preserves_logical_outcome_evidence(
        self, mock_registry, mock_persist
    ):
        result = {
            "status": "completed",
            "outcome": "partial",
            "summary": "usable but incomplete",
            "interrupted": False,
            "tool_error_count": 2,
            "exit_reason": "max_iterations",
            "api_calls": 50,
            "duration_seconds": 12.0,
        }

        _push_completion_event(
            {
                "delegation_id": "single-1",
                "dispatched_at": 1.0,
                "completed_at": 13.0,
            },
            result,
            "completed",
        )

        event = mock_registry.completion_queue.put.call_args.args[0]
        self.assertEqual(event["outcome"], "partial")
        self.assertFalse(event["interrupted"])
        self.assertEqual(event["tool_error_count"], 2)
        self.assertEqual(event["exit_reason"], "max_iterations")
        mock_persist.assert_called_once_with(event, result)


class TestFormatterNeutralisation(unittest.TestCase):
    def _batch_evt(self, result):
        return {
            "type": "async_delegation",
            "delegation_id": "d1",
            "is_batch": True,
            "goals": ["accomplish X"],
            "results": [result],
        }

    def test_unverified_renders_warning_not_success_icon(self):
        out = _format_async_delegation(self._batch_evt({
            "task_index": 0,
            "status": "completed",
            "outcome": "unverified",
            "summary": "I uploaded the file successfully.",
        }))
        # The child's own summary is still shown...
        self.assertIn("I uploaded the file successfully.", out)
        # ...but never with a green success checkmark.
        self.assertNotIn("✓", out)
        # And the parent is explicitly told finishing != acceptance.
        self.assertIn("verif", out.lower())

    def test_partial_output_visible_but_not_success(self):
        out = _format_async_delegation(self._batch_evt({
            "task_index": 0,
            "status": "completed",
            "outcome": "partial",
            "summary": "Reached step 3 of 5 before stopping.",
        }))
        self.assertIn("Reached step 3 of 5 before stopping.", out)
        self.assertNotIn("✓", out)

    def test_failed_output_visible_but_not_success(self):
        out = _format_async_delegation(self._batch_evt({
            "task_index": 0,
            "status": "failed",
            "outcome": "failed",
            "summary": None,
            "error": "boom",
        }))
        self.assertNotIn("✓", out)
        self.assertIn("boom", out)

    def test_legacy_result_without_outcome_fails_closed(self):
        # A legacy child result predating the `outcome` field: status=completed
        # with a summary must NOT be rendered as green success.
        out = _format_async_delegation(self._batch_evt({
            "task_index": 0,
            "status": "completed",
            "summary": "legacy summary text",
        }))
        self.assertNotIn("✓", out)
        self.assertIn("legacy summary text", out)
        self.assertIn("verif", out.lower())

    def test_authoritative_error_overrides_explicit_partial_outcome(self):
        result = {
            "status": "completed",
            "outcome": "partial",
            "summary": "partial diagnostic output",
            "error": "runtime crashed",
        }
        self.assertEqual(_derive_result_outcome(result), "failed")
        out = _format_async_delegation(self._batch_evt(result))
        self.assertIn("outcome=failed", out)
        self.assertIn("runtime crashed", out)

    def test_fatal_exit_reason_overrides_explicit_unverified_outcome(self):
        result = {
            "status": "completed",
            "outcome": "unverified",
            "summary": "restart required",
            "exit_reason": "code_skew_attribute_error(missing member)",
        }
        self.assertEqual(_derive_result_outcome(result), "failed")

    def test_legacy_error_and_fatal_exit_reason_fail_closed(self):
        self.assertEqual(
            _derive_result_outcome({
                "status": "completed",
                "summary": "legacy diagnostic",
                "error": "runtime crashed",
            }),
            "failed",
        )
        self.assertEqual(
            _derive_result_outcome({
                "status": "completed",
                "summary": "legacy restart diagnostic",
                "exit_reason": "code_skew_detected",
            }),
            "failed",
        )

    def test_runtime_evidence_is_injected_for_parent_verification(self):
        out = _format_async_delegation(self._batch_evt({
            "task_index": 0,
            "status": "completed",
            "outcome": "partial",
            "summary": "usable but incomplete",
            "exit_reason": "max_iterations",
            "interrupted": False,
            "tool_error_count": 2,
        }))
        self.assertIn("exit_reason=max_iterations", out)
        self.assertIn("interrupted=false", out)
        self.assertIn("tool_errors=2", out)

    def test_legacy_unknown_status_remains_unknown(self):
        self.assertEqual(
            _derive_result_outcome({"status": "unknown", "summary": None}),
            "unknown",
        )

    def test_legacy_max_iteration_exit_is_partial_with_usable_output(self):
        for reason in ("max_iterations", "max_iterations_reached(50/50)"):
            with self.subTest(reason=reason):
                self.assertEqual(
                    _derive_result_outcome({
                        "status": "completed",
                        "summary": "legacy partial progress",
                        "exit_reason": reason,
                    }),
                    "partial",
                )

    def test_partial_exit_evidence_overrides_explicit_unverified_outcome(self):
        self.assertEqual(
            _derive_result_outcome({
                "status": "completed",
                "outcome": "unverified",
                "summary": "recovered partial stream",
                "exit_reason": "partial_stream_recovery",
            }),
            "partial",
        )

    def test_legacy_max_iteration_without_output_is_failed(self):
        self.assertEqual(
            _derive_result_outcome({
                "status": "completed",
                "summary": None,
                "exit_reason": "max_iterations",
            }),
            "failed",
        )

    def test_abandoned_recovery_diagnostic_preserves_unknown_outcome(self):
        result = {
            "status": "unknown",
            "outcome": "unknown",
            "summary": None,
            "error": "Delegation owner exited; outcome unknown.",
            "error_authoritative": False,
        }
        self.assertEqual(_derive_result_outcome(result), "unknown")

    def test_unmarked_error_remains_authoritative_for_compatibility(self):
        self.assertEqual(
            _derive_result_outcome({
                "status": "unknown",
                "outcome": "unknown",
                "error": "runtime crashed",
            }),
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
