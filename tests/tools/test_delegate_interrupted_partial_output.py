"""Invariant: a child stopped mid-work reports what it HAD so far. The loop's ``final_response`` for an
interrupted turn is the placeholder ``"Operation interrupted."`` (also appended as the closing assistant row);
the parent-visible entry must carry the child's last real assistant text as ``summary`` and keep the
placeholder as ``error`` — never a partial result that reads as an empty stop. See #114456.
"""
from types import SimpleNamespace

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


def test_interrupted_child_entry_carries_its_partial_output():
    messages = [
        {"role": "user", "content": "kickoff"},
        {"role": "assistant", "content": [{"type": "text", "text": "Audited 3 of 7 modules; two findings so far."}],
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "[Command interrupted]"},
        {"role": "assistant", "content": "Operation interrupted."},
    ]
    result = {"final_response": "Operation interrupted.", "messages": messages, "api_calls": 2,
              "completed": False, "interrupted": True}
    child = SimpleNamespace(model="m", session_estimated_cost_usd=0.0, session_cost_status="unknown",
                            session_prompt_tokens=1, session_completion_tokens=1, _delegate_role="leaf")

    entry = _build_result_entry(child, result, 0, 12.5, _SchemaOutcome(None, None, [], 0))

    assert entry["status"] == entry["exit_reason"] == "interrupted"
    assert entry["summary"] == "Audited 3 of 7 modules; two findings so far."
    assert entry["error"] == "Operation interrupted."
