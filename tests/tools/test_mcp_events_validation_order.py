"""MCP event handlers run validation and the consent gate before any transport work.

Subscribing installs a standing trigger that starts unattended agent runs, so
``subscribe_event`` fails closed: the user is asked every time (``_consent_error``).
That gate — and every other argument check — must run before
``tools.mcp_tool_discovery._get_connected_server_for_call`` (the single first-use
connect point for lazy servers, i.e. acquiring it spawns the server): a declined
call must not spawn/connect the server first, and a rejected argument must not
reach for a server at all.
"""

import json

import pytest

from tools.mcp_tool_events import _make_subscribe_event_handler


@pytest.fixture
def acquisition(monkeypatch):
    """Transport-boundary double: records every connection attempt, never connects."""
    calls = []

    def fake_get(server_name):
        calls.append(server_name)
        return None  # "not connected": keeps the handler's post-acquisition path observable

    monkeypatch.setattr("tools.mcp_tool_discovery._get_connected_server_for_call", fake_get)
    return calls


class TestValidationAndConsentBeforeTransport:
    @pytest.mark.parametrize("args,expected_error", [
        ({"deliver": "log"}, "Missing required parameter 'name'"),
        ({"name": "x", "arguments": []}, "'arguments' must be an object"),
        ({"name": "x", "deliver_chat_id": 123}, "'deliver_chat_id' must be a string"),
    ], ids=["missing-name", "arguments-not-object", "deliver-chat-id-not-string"])
    def test_rejected_arguments_never_connect(self, acquisition, args, expected_error):
        handler = _make_subscribe_event_handler("ambush", 5.0)
        out = json.loads(handler(dict(args)))
        assert out["error"] == expected_error
        assert acquisition == []

    def test_declined_consent_never_connects(self, acquisition, monkeypatch):
        monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent",
                            lambda *a, **k: "decline")
        handler = _make_subscribe_event_handler("ambush", 5.0)
        out = json.loads(handler({"name": "feed.emission.created", "deliver": "log"}))
        assert "did not approve" in out["error"]
        assert acquisition == []

    def test_accepted_consent_still_connects(self, acquisition, monkeypatch):
        monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent",
                            lambda *a, **k: "accept")
        handler = _make_subscribe_event_handler("ambush", 5.0)
        out = json.loads(handler({"name": "feed.emission.created", "deliver": "log"}))
        assert "is not connected" in out["error"]
        assert acquisition == ["ambush"]
