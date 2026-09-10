import json

from tools.send_message_tool import SEND_MESSAGE_SCHEMA, send_message_tool


def _params():
    return SEND_MESSAGE_SCHEMA["parameters"]


def test_send_message_schema_keeps_action_optional_for_default_send():
    assert _params()["required"] == []


def test_send_message_schema_requires_delivery_fields_unless_listing_targets():
    assert _params()["anyOf"] == [
        {"properties": {"action": {"const": "list"}}, "required": ["action"]},
        {"required": ["target", "message"]},
    ]


def test_send_message_list_action_still_accepts_no_target_or_message():
    result = json.loads(send_message_tool({"action": "list"}))

    assert "targets" in result
    assert "error" not in result
