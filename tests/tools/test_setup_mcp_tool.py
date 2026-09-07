"""setup_mcp tool — the desktop inline MCP consent card's tool half.

Behavior contracts:
- no callback (not the desktop app) → tool_error pointing at the CLI path
- empty/invalid args → tool_error
- callback answer passes through as JSON
- empty callback answer (timeout) → status "unanswered", never an error
"""

import json

import pytest

from tools.setup_mcp_tool import SETUP_MCP_SCHEMA, setup_mcp_tool


def test_schema_forbids_hand_editing_mcp_servers_config():
    # Nothing else teaches the model this: the tool is desktop_ui-only, so
    # without it a model could just write_file into mcp_servers config
    # directly, bypassing the consent-card/OAuth flow this tool exists for.
    assert "hand-edit" in SETUP_MCP_SCHEMA["description"]
    assert "mcp_servers" in SETUP_MCP_SCHEMA["description"]


def test_requires_desktop_callback():
    result = json.loads(setup_mcp_tool(server="linear", callback=None))
    assert "error" in result
    assert "hermes mcp install" in result["error"]


def test_requires_server_name():
    result = json.loads(setup_mcp_tool(server="  ", callback=lambda *a: ""))
    assert "error" in result


def test_rejects_unknown_action():
    result = json.loads(
        setup_mcp_tool(server="linear", action="uninstall", callback=lambda *a: "")
    )
    assert "error" in result
    assert "action" in result["error"]


def test_passes_through_renderer_outcome():
    outcome = {"status": "installed", "server": "linear"}

    def cb(server, action, reason):
        assert server == "linear"
        assert action == "install"
        assert reason == "to read tickets"
        return json.dumps(outcome)

    result = json.loads(
        setup_mcp_tool(server="linear", action="install", reason="to read tickets", callback=cb)
    )
    assert result == outcome


def test_timeout_returns_unanswered_not_error():
    result = json.loads(setup_mcp_tool(server="figma", callback=lambda *a: ""))
    assert result["status"] == "unanswered"
    assert result["server"] == "figma"


def test_callback_exception_is_tool_error():
    def cb(*a):
        raise RuntimeError("gateway went away")

    result = json.loads(setup_mcp_tool(server="figma", callback=cb))
    assert "error" in result


def test_non_json_answer_wrapped_as_error_status():
    result = json.loads(setup_mcp_tool(server="figma", callback=lambda *a: "garbage"))
    assert result["status"] == "error"


@pytest.mark.parametrize("action", ["install", "enable", "authorize"])
def test_all_actions_accepted(action):
    result = json.loads(
        setup_mcp_tool(server="x", action=action, callback=lambda s, a, r: json.dumps({"status": "declined", "server": s}))
    )
    assert result["status"] == "declined"
