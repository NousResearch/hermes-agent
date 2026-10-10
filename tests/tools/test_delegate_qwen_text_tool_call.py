"""Regression coverage for Qwen XML tool calls leaked into delegate final text (#128999)."""

from collections import deque

from tools.delegate_tool import _run_single_child
from tools.delegate_tool_child_run import _qwen_text_tool_call_name, _retry_leaked_qwen_tool_call


QWEN_WRITE = """I will update the file now.
<tool_call>
<function=write_file>
<parameter=path>a.py</parameter>
<parameter=content>print("ok")</parameter>
</function>
</tool_call>"""

QWEN_READ = """<tool_call>
<function=read_file>
<parameter=path>a.py</parameter>
</function>
</tool_call>"""

QWEN_JSON_READ = """I will inspect the file first.
<tool_call>
{"name":"read_file","arguments":{"path":"a.py"}}
</tool_call>"""


class _DelegateChildDouble:
    def __init__(self, replies):
        self._replies = deque(replies)
        self.prompts = []
        self.valid_tool_names = frozenset(("write_file", "read_file"))
        self.tool_progress_callback = None
        self._delegate_saved_tool_names = ()
        self._credential_pool = None
        self._subagent_id = None
        self._delegate_depth = 1
        self._parent_subagent_id = None
        self._delegate_output_schema = None
        self.model = "test-model"
        self.session_id = "qwen-leak-test"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_reasoning_tokens = 0

    @staticmethod
    def get_activity_summary():
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.prompts.append(user_message)
        reply = self._replies.popleft()
        if isinstance(reply, dict):
            payload = {**reply}
            payload.setdefault("api_calls", 1)
            payload.setdefault("messages", [])
            return payload
        return {"final_response": reply, "completed": True, "api_calls": 1, "messages": []}

    @staticmethod
    def close():
        return None


class _DelegateParentDouble:
    _current_task_id = None
    _delegate_depth = 0

    @staticmethod
    def _touch_activity(_description):
        return None


def _execute(child):
    return _run_single_child(0, "update a.py", child, _DelegateParentDouble())


def test_detects_both_qwen_envelopes_after_narration():
    child = _DelegateChildDouble(())
    assert _qwen_text_tool_call_name(child, QWEN_WRITE) == "write_file"
    assert _qwen_text_tool_call_name(child, QWEN_JSON_READ) == "read_file"


def test_marker_discussion_and_unavailable_tool_are_not_execution():
    child = _DelegateChildDouble(())
    explanatory = "The <tool_call> marker is followed by prose about <function=write_file> in parser docs."
    unavailable = "<tool_call>\n<function=deploy_to_prod>\n</function>\n</tool_call>"
    assert _qwen_text_tool_call_name(child, explanatory) is None
    assert _qwen_text_tool_call_name(child, unavailable) is None


def test_correction_turn_replaces_leaked_action_with_real_summary():
    child = _DelegateChildDouble((QWEN_WRITE, "Done: a.py was updated and verified."))
    entry = _execute(child)
    assert len(child.prompts) == 2
    assert "was not executed" in child.prompts[-1]
    assert (entry["status"], entry["exit_reason"]) == ("completed", "completed")
    assert entry["summary"] == "Done: a.py was updated and verified."


def test_second_leak_fails_through_existing_result_contract():
    child = _DelegateChildDouble((QWEN_WRITE, QWEN_READ))
    entry = _execute(child)
    assert len(child.prompts) == 2
    assert (entry["status"], entry["exit_reason"], entry["truncated"]) == ("failed", "error", False)
    assert entry["failure_reason"] == "unparsed_tool_call"
    assert entry["summary"] == QWEN_READ


def test_provider_failure_during_correction_skips_schema_retry_and_stays_failed():
    child = _DelegateChildDouble((
        QWEN_WRITE,
        {
            "final_response": "provider rejected the correction request",
            "completed": False,
            "failed": True,
            "error": "provider rejected the correction request",
            "failure_reason": "provider_error",
        },
    ))
    # The correction failure is terminal even when a schema was requested.
    # Without the guard in _validate_child_output_schema this would spend a
    # third child turn trying to validate/rewrite the provider error text.
    child._delegate_output_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    entry = _execute(child)
    assert (entry["status"], entry["exit_reason"]) == ("failed", "error")
    assert entry["failure_reason"] == "provider_error"
    assert entry["schema_valid"] is False
    assert len(child.prompts) == 2


def test_interrupt_short_circuits_before_any_correction():
    child = _DelegateChildDouble(("unused",))
    interrupted = {"final_response": QWEN_WRITE, "interrupted": True, "api_calls": 1, "messages": []}
    triggered = _retry_leaked_qwen_tool_call(child, interrupted, 0, "child-0", None)
    assert triggered is False
    assert not child.prompts


def test_rescued_text_is_still_checked_by_declared_schema():
    child = _DelegateChildDouble((QWEN_WRITE, '{"city": "Oslo"}'))
    child._delegate_output_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    entry = _execute(child)
    assert entry["schema_valid"] is True
    assert entry["summary"] == '{"city": "Oslo"}'
    assert len(child.prompts) == 2
