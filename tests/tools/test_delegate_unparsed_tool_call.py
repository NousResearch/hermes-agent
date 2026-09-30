"""delegate_task never accepts an unparsed <tool_call>/<function= block as work (#128999).

Local-stack models (qwen behind sglang) intermittently emit the tool call as assistant
TEXT when their chat template rejects it. The leaf runner used to treat that text as the
final answer: ``status=completed`` with zero work done. Contract now: one bounded
correction retry (same posture as the typed schema-reject retry), then — still unparsed —
the run is a PARSE FAILURE, never a completion. Prose that merely MENTIONS the markers
never triggers the path.
"""

from tools.delegate_tool import _run_single_child
from tools.delegate_tool_child_run import _is_unparsed_tool_call_text

TOOL_CALL_BLOCK = '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py"}}\n</tool_call>'
FUNCTION_BLOCK = '<function=read_file>("a.py")</function>'


class _StubChild:
    """Minimal child agent double (mirrors test_delegate_output_schema)."""

    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None  # skip registry
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    model = "test-model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list = []

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.calls.append(user_message)
        text = self.responses.pop(0)
        return {
            "final_response": text,
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


def _run(child):
    return _run_single_child(0, "patch the build file", child, _StubParent())


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def test_whole_answer_tool_call_block_is_unparsed():
    assert _is_unparsed_tool_call_text(TOOL_CALL_BLOCK) is True
    assert _is_unparsed_tool_call_text("  \n" + TOOL_CALL_BLOCK + "\n") is True


def test_function_form_and_markdown_fenced_block_are_unparsed():
    assert _is_unparsed_tool_call_text(FUNCTION_BLOCK) is True
    assert _is_unparsed_tool_call_text("```xml\n" + TOOL_CALL_BLOCK + "\n```") is True


def test_prose_mentioning_the_markers_is_not_unparsed():
    assert _is_unparsed_tool_call_text(
        "The qwen template rejects the <tool_call> tag, so the call lands as text; I worked "
        "around it by editing the file directly. Done.") is False
    assert _is_unparsed_tool_call_text("Task finished: patch applied and tests pass.") is False


def test_detector_is_fail_closed_on_non_text():
    assert _is_unparsed_tool_call_text("") is False
    assert _is_unparsed_tool_call_text(None) is False
    assert _is_unparsed_tool_call_text(["<tool_call>"]) is False


# ---------------------------------------------------------------------------
# Runner: bounded retry, then parse failure — never a fake completion
# ---------------------------------------------------------------------------

def test_unparsed_block_retried_once_then_prose_completes():
    child = _StubChild([TOOL_CALL_BLOCK, "Done: patched the build file and tests pass."])
    entry = _run(child)
    # exactly one correction turn, and it explains WHY nothing ran
    assert len(child.calls) == 2
    assert "plain text" in child.calls[1] and "NOT done" in child.calls[1]
    assert entry["status"] == "completed"
    assert entry["exit_reason"] == "completed"
    assert entry["summary"] == "Done: patched the build file and tests pass."


def test_unparsed_block_twice_is_a_parse_failure_never_completed():
    child = _StubChild([TOOL_CALL_BLOCK, FUNCTION_BLOCK])
    entry = _run(child)
    assert len(child.calls) == 2  # bounded: exactly one retry
    assert entry["status"] == "failed"
    assert entry["exit_reason"] == "unparsed_tool_call"
    assert entry["failure_reason"] == "unparsed_tool_call"
    assert "no tool ran" in entry["error"]
    assert entry["truncated"] is False
    # the raw block rides along for the parent's diagnosis
    assert entry["summary"] == FUNCTION_BLOCK


def test_prose_answer_mentioning_markers_takes_no_retry_turn():
    child = _StubChild([
        "The <tool_call> tag was rejected by the template, so I edited the file directly. Done."])
    entry = _run(child)
    assert len(child.calls) == 1  # no correction turn for legitimate prose
    assert entry["status"] == "completed"


def test_interrupted_child_skips_the_retry_turn():
    class _InterruptedChild(_StubChild):
        def run_conversation(self, user_message, task_id=None, **_kwargs):
            out = super().run_conversation(user_message, task_id)
            out["interrupted"] = True
            return out

    child = _InterruptedChild([TOOL_CALL_BLOCK])
    entry = _run(child)
    assert len(child.calls) == 1  # interrupted children never get the correction turn
    assert entry["status"] == "interrupted"
