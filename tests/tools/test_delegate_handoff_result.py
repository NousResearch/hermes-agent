import json

from tools.delegate_tool import HandoffResult, _handoff_error, _run_single_child


class FakeParent:
    _current_task_id = "parent-task"
    _active_children = []
    _active_children_lock = None

    def _touch_activity(self, desc):
        self.last_activity = desc


class FakeChild:
    _delegate_role = "leaf"
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_saved_tool_names = []
    model = "test-model"
    session_prompt_tokens = 11
    session_completion_tokens = 7
    session_estimated_cost_usd = 0.0123
    session_reasoning_tokens = 0
    session_id = "child-session"
    tool_progress_callback = None

    def __init__(self, result):
        self.result = result
        self.closed = False

    def run_conversation(self, **kwargs):
        return self.result

    def get_activity_summary(self):
        return {
            "current_tool": None,
            "api_call_count": 1,
            "max_iterations": 3,
            "last_activity_desc": "testing",
        }

    def close(self):
        self.closed = True


def test_handoff_result_serializes_typed_frame_with_legacy_keys():
    frame = HandoffResult.from_child_run(
        task_index=2,
        status="completed",
        summary="done",
        exit_reason="completed",
        api_calls="3",
        duration_seconds="1.5",
        model="m",
        tokens={"input": 1, "output": 2},
        tool_trace=[{"tool": "read_file"}],
        child_role="leaf",
        child_cost_usd="0.25",
    ).to_dict()

    assert frame["type"] == "handoff_result"
    assert frame["handoff_schema"] == "delegate.handoff_result.v1"
    assert frame["task_index"] == 2
    assert frame["status"] == "completed"
    assert frame["summary"] == "done"
    assert frame["api_calls"] == 3
    assert frame["duration_seconds"] == 1.5
    assert frame["tokens"] == {"input": 1, "output": 2}
    assert frame["tool_trace"] == [{"tool": "read_file"}]
    assert frame["_child_role"] == "leaf"
    assert frame["_child_cost_usd"] == 0.25
    json.dumps(frame)


def test_handoff_error_uses_formal_schema_and_exit_reason():
    frame = _handoff_error(
        task_index=0,
        status="timeout",
        error="stuck",
        exit_reason="timeout",
        duration_seconds=9,
        child_role="orchestrator",
        diagnostic_path="/tmp/diag.txt",
    )

    assert frame["type"] == "handoff_result"
    assert frame["handoff_schema"] == "delegate.handoff_result.v1"
    assert frame["summary"] is None
    assert frame["error"] == "stuck"
    assert frame["exit_reason"] == "timeout"
    assert frame["diagnostic_path"] == "/tmp/diag.txt"
    assert frame["_child_role"] == "orchestrator"


def test_run_single_child_returns_formal_handoff_result():
    child = FakeChild(
        {
            "final_response": "implemented",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }
    )

    frame = _run_single_child(0, "do work", child=child, parent_agent=FakeParent())

    assert frame["type"] == "handoff_result"
    assert frame["handoff_schema"] == "delegate.handoff_result.v1"
    assert frame["task_index"] == 0
    assert frame["status"] == "completed"
    assert frame["summary"] == "implemented"
    assert frame["exit_reason"] == "completed"
    assert frame["api_calls"] == 1
    assert frame["model"] == "test-model"
    assert frame["tokens"] == {"input": 11, "output": 7}
    assert frame["_child_role"] == "leaf"
    assert frame["_child_cost_usd"] == 0.0123
    assert child.closed is True
