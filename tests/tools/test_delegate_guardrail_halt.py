import unittest
from tools.delegate_tool import _run_single_child


class _StubChild:
    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    _tool_guardrail_halt_decision = None
    model = "test-model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, response_dict):
        self.response_dict = response_dict

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        return dict(self.response_dict)

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


class TestDelegateGuardrailHalt(unittest.TestCase):
    def test_guardrail_halt_in_result_reported_as_failed(self):
        guardrail_meta = {
            "action": "halt",
            "code": "loop_web_search_cap",
            "tool_name": "web_search",
            "count": 15,
        }
        child = _StubChild({
            "final_response": "Search limit reached; stopped.",
            "completed": True,
            "turn_exit_reason": "guardrail_halt",
            "guardrail": guardrail_meta,
            "api_calls": 15,
            "messages": [],
        })
        entry = _run_single_child(0, "research topic", child, _StubParent())
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["exit_reason"], "guardrail_halt")
        self.assertIn("loop_web_search_cap", entry.get("error", ""))
        self.assertEqual(entry.get("guardrail"), guardrail_meta)
        self.assertEqual(entry["summary"], "Search limit reached; stopped.")

    def test_guardrail_halt_on_child_attribute_reported_as_failed(self):
        class _Decision:
            def to_metadata(self):
                return {
                    "action": "halt",
                    "code": "loop_web_search_cap",
                    "tool_name": "web_search",
                }

        child = _StubChild({
            "final_response": "Stopped by guardrail.",
            "completed": True,
            "api_calls": 5,
            "messages": [],
        })
        child._tool_guardrail_halt_decision = _Decision()
        entry = _run_single_child(0, "research topic", child, _StubParent())
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["exit_reason"], "guardrail_halt")
        self.assertIn("loop_web_search_cap", entry.get("error", ""))
        self.assertEqual(entry.get("guardrail", {}).get("code"), "loop_web_search_cap")

    def test_normal_completion_retains_completed_status(self):
        child = _StubChild({
            "final_response": "Task finished successfully.",
            "completed": True,
            "api_calls": 2,
            "messages": [],
        })
        entry = _run_single_child(0, "simple task", child, _StubParent())
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["exit_reason"], "completed")
        self.assertNotIn("error", entry)
        self.assertNotIn("guardrail", entry)

    def test_max_iterations_without_guardrail_retains_classification(self):
        child = _StubChild({
            "final_response": "Partial summary of work done.",
            "completed": False,
            "api_calls": 5,
            "messages": [],
        })
        entry = _run_single_child(0, "heavy task", child, _StubParent())
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["exit_reason"], "max_iterations")
        self.assertTrue(entry.get("truncated"))
        self.assertNotIn("guardrail", entry)
