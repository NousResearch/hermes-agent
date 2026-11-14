"""A resumed transcript keeps what the user answered to a ``clarify`` question."""

import json

from tui_gateway import server


def test_clarify_tool_row_keeps_its_result_for_resume():
    result = json.dumps({
        "responses": [{"question": "Which colour?", "status": "answered", "user_response": "red"}],
        "outcome": "submitted",
    })
    history = [
        {"role": "user", "content": "paint it"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "clarify", "arguments": json.dumps({"questions": [{"question": "Which colour?"}]})},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": result},
    ]

    tool_rows = [m for m in server._history_to_messages(history) if m["role"] == "tool"]

    assert tool_rows[0]["name"] == "clarify"
    assert json.loads(tool_rows[0]["content"])["responses"][0]["user_response"] == "red"
