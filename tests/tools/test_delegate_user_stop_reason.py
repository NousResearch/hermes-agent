"""Invariant: a child that was stopped ON PURPOSE (Stop button / /agents overlay / subagent.interrupt RPC, or the
parent model's delegate_task(action='stop')) names the cause in its result entry. A bare ``interrupted`` reads
as a failure to the parent, which then re-dispatches the very task that was just cancelled (kilocode#14701:
7/7 re-dispatches without the cause, 0/5 with it). The stamp lives on the child agent, not the interrupt state,
because the loop clears that state on every exit path.
"""
from types import SimpleNamespace

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry
from tools.delegate_tool_registry import request_subagent_stop


class _Child(SimpleNamespace):
    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.stops = getattr(self, "stops", []) + [message]


def _entry(child):
    result = {"final_response": "Operation interrupted.", "messages": [
        {"role": "user", "content": "kickoff"},
        {"role": "assistant", "content": "Iteration 1 done."},
        {"role": "assistant", "content": "Operation interrupted."},
    ], "api_calls": 2, "completed": False, "interrupted": True}
    return _build_result_entry(child, result, 0, 3.0, _SchemaOutcome(None, None, [], 0))


def _child():
    return _Child(model="m", session_estimated_cost_usd=0.0, session_cost_status="unknown",
                  session_prompt_tokens=1, session_completion_tokens=1, _delegate_role="leaf")


def test_deliberate_stop_is_named_in_the_entry_and_a_plain_interrupt_is_not():
    child = _child()
    plain = _entry(child)
    assert plain["status"] == "interrupted" and "stopped_by" not in plain
    assert plain["error"] == "Operation interrupted."

    assert request_subagent_stop(child, "sa-0-abc") is True
    assert child.stops == ["Interrupted via TUI (sa-0-abc)"]
    stopped = _entry(child)
    assert stopped["stopped_by"] == "user"
    assert stopped["summary"] == "Iteration 1 done."
    assert stopped["error"].startswith("Stopped by the user")
    assert "do not re-dispatch" in stopped["error"]

    by_parent = _child()
    request_subagent_stop(by_parent, "sa-1-def", by="parent")
    entry = _entry(by_parent)
    assert entry["stopped_by"] == "parent"
    assert entry["error"].startswith("Stopped by you (delegate_task action='stop')")
