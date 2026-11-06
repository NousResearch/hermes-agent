"""Delegated execution evidence must distinguish process success from task proof."""
import json
from types import SimpleNamespace

import pytest

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry, _build_tool_trace
from tools.delegate_tool_results import _apply_summary_budget


def entry_for(envelopes, *, summary="Checked the requested cases; all passed.", **run_state):
    messages = []
    for index, envelope in enumerate(envelopes):
        call_id = f"check-{index}"
        messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps({"command": "test-command"})}}]})
        if envelope is not None:
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(envelope)})
    child = SimpleNamespace(model="fixture", session_estimated_cost_usd=0,
        session_cost_status="unknown", session_prompt_tokens=100, session_completion_tokens=20,
        _delegate_role="leaf")
    return _build_result_entry(child, {"final_response": summary, "messages": messages,
        "completed": True, **run_state}, 0, 1, _SchemaOutcome(None, None, [], 0))


def test_completed_check_returns_observed_execution_without_repeating_output():
    result = entry_for([{"exit_code": 0, "output": "37 assertions passed", "error": None}])
    assert result["status"] == result["exit_reason"] == "completed"
    assert result["truncated"] is False
    assert result["tool_trace"][0]["execution"] == {
        "outcome": "succeeded", "exit_code": 0, "output_truncated": False}
    assert "37 assertions passed" not in json.dumps(result["tool_trace"])
    assert "verified" not in result  # A completed command is not an acceptance verdict.


@pytest.mark.parametrize("envelope,expected", [
    ({"exit_code": 1, "output": "assertion failed", "error": None}, "failed"),
    ({"exit_code": 0, "output": "no check ran", "error": "execution rejected"}, "failed"),
    ({"exit_code": 0, "output": "no check ran", "status": "failed"}, "failed"),
    ({"exit_code": None, "output": "still running", "session_id": "job"}, "unknown"),
    ({"exit_code": 0, "output": "", "session_id": "job"}, "unknown"),
    ({"exit_code": 124, "output": "", "error": None}, "unknown"),
    ({"exit_code": 0, "output": "[Command timed out after 30s]"}, "unknown"),
    ({"exit_code": True, "output": "all good"}, "unknown"),
    ({"output": "success claimed without a process outcome"}, "unknown"),
])
def test_success_summary_does_not_override_failed_or_incomplete_execution(envelope, expected):
    result = entry_for([envelope], summary="Everything passed.")
    trace = result["tool_trace"][0]
    assert result["summary"] == "Everything passed."
    assert trace["execution"]["outcome"] == expected
    if expected == "failed":
        assert trace["status"] == "error"


def test_missing_tool_result_or_tools_cannot_establish_execution():
    result = entry_for([None], summary="Done.")
    assert "execution" not in result["tool_trace"][0]
    assert entry_for([], summary="Done.")["tool_trace"] == []


@pytest.mark.parametrize("result_id", [None, "unrelated-call"])
def test_unmatched_result_cannot_establish_execution_for_latest_call(result_id):
    messages = [{"role": "assistant", "tool_calls": [{"id": "check", "type": "function",
        "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": result_id,
         "content": json.dumps({"exit_code": 0, "output": "passed"})}]
    trace = _build_tool_trace(messages)
    assert len(trace) == 1
    assert "execution" not in trace[0]


@pytest.mark.parametrize("truncation", [
    {"output_total_chars": 9000}, {"truncation_note": "more in saved output"}, {"truncated": True},
])
def test_successful_execution_keeps_output_truncation_explicit(truncation):
    result = entry_for([{"exit_code": 0, "output": "37 checks...", "error": None, **truncation}])
    evidence = result["tool_trace"][0]["execution"]
    assert evidence["outcome"] == "succeeded"
    assert evidence["output_truncated"] is True


def test_failed_check_remains_visible_beside_later_success_and_summary(monkeypatch):
    result = entry_for([{"exit_code": 1, "output": "failed case", "error": None},
                        {"exit_code": 0, "output": "other case passed", "error": None}])
    assert [item["execution"]["outcome"] for item in result["tool_trace"]] == ["failed", "succeeded"]
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"max_summary_chars": 2000})
    result["summary"] = "checked result\n" * 1000
    _apply_summary_budget([result], None)
    assert result["summary_truncated"] is True
    assert result["tool_trace"][0]["execution"]["outcome"] == "failed"


@pytest.mark.parametrize("state", [{"completed": False}, {"failed": True, "error": "provider stopped"},
                                   {"interrupted": True, "completed": False}])
def test_successful_command_does_not_erase_incomplete_child_state(state):
    result = entry_for([{"exit_code": 0, "output": "one check passed", "error": None}], **state)
    assert result["tool_trace"][0]["execution"]["outcome"] == "succeeded"
    assert result["exit_reason"] != "completed"
    if state.get("failed"):
        assert result["status"] == "failed"
    elif state.get("interrupted"):
        assert result["status"] == "interrupted"
    else:
        assert result["truncated"] is True
