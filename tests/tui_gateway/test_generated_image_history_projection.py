import json

from tui_gateway.server import _history_to_messages


def _image_turn():
    result = {"success": True, "image": "/tmp/cache/generated/images/fox.png", "model": "gpt-image-2"}
    return [
        {"role": "user", "content": "draw a fox"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_img", "type": "function", "function": {
            "name": "image_generate", "arguments": json.dumps({"prompt": "a fox"})}}]},
        {"role": "tool", "tool_call_id": "call_img", "content": json.dumps(result)},
        {"role": "assistant", "content": "Here is your fox."},
    ], result


def test_compact_projection_keeps_the_generated_image_result():
    history, result = _image_turn()
    tool_rows = [m for m in _history_to_messages(history) if m.get("role") == "tool"]
    assert len(tool_rows) == 1 and tool_rows[0]["name"] == "image_generate"
    assert json.loads(tool_rows[0]["content"]) == result


def test_other_tool_outputs_stay_omitted():
    history = [
        {"role": "user", "content": "list it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "terminal", "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "secret.txt"},
        {"role": "assistant", "content": "Done."},
    ]
    tool_rows = [m for m in _history_to_messages(history) if m.get("role") == "tool"]
    assert tool_rows and "content" not in tool_rows[0]
