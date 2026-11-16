"""``send_message_tool`` sends and does nothing else: any other ``action`` is refused before a send."""

import json
from unittest.mock import patch

import pytest

from tools.send_message_tool import send_message_tool


@pytest.mark.parametrize("action", [None, "Send", "list", "react", "unreact", "bogus"])
def test_any_action_but_send_is_refused_and_nothing_is_sent(action):
    with patch("tools.send_message_tool._handle_send") as send:
        result = json.loads(send_message_tool({"action": action, "target": "telegram:1", "message": "hi"}))

    assert "Unknown action" in result["error"]
    send.assert_not_called()


@pytest.mark.parametrize("args", [{}, {"action": "send"}])
def test_send_and_a_call_without_an_action_both_send(args):
    with patch("tools.send_message_tool._handle_send", return_value='{"success": true}') as send:
        result = json.loads(send_message_tool({**args, "target": "telegram:1", "message": "hi"}))

    assert result == {"success": True}
    send.assert_called_once()
