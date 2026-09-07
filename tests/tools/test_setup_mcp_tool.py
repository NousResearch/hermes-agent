"""setup_mcp tool — the desktop inline connector card's tool half.

Behavior contracts:
- no callback (not the desktop app) → tool_error pointing at the CLI path
- empty/invalid args → tool_error
- singular `server` and list `servers` merge into one ordered, deduped list
- the answer always reaches the agent with a per-connector breakdown, whether
  the renderer sent the flat single-connector shape or the list shape
- empty callback answer (timeout) → status "unanswered", never an error
"""

import json

import pytest

from tools.setup_mcp_tool import _MAX_CONNECTORS, setup_mcp_tool


def _echo(payload):
    """Callback that records what the tool asked for and answers with it."""
    seen = {}

    def cb(servers, action, reason):
        seen["servers"] = servers
        seen["action"] = action
        seen["reason"] = reason
        return json.dumps(payload)

    return cb, seen


def test_requires_desktop_callback():
    result = json.loads(setup_mcp_tool(server="linear", callback=None))
    assert "error" in result
    assert "hermes mcp install" in result["error"]


def test_requires_a_connector_name():
    result = json.loads(setup_mcp_tool(server="  ", callback=lambda *a: ""))
    assert "error" in result


def test_rejects_unknown_action():
    result = json.loads(
        setup_mcp_tool(server="linear", action="uninstall", callback=lambda *a: "")
    )
    assert "error" in result
    assert "action" in result["error"]


def test_passes_through_renderer_outcome():
    cb, seen = _echo({"status": "installed", "server": "linear"})

    result = json.loads(
        setup_mcp_tool(server="linear", action="install", reason="to read tickets", callback=cb)
    )

    assert seen == {"servers": ["linear"], "action": "install", "reason": "to read tickets"}
    assert result["status"] == "installed"
    # A flat single-connector answer is lifted into the list contract so the
    # agent reads one shape no matter how many connectors it asked for.
    assert result["connectors"] == [{"server": "linear", "status": "installed"}]


def test_defaults_to_connect_so_the_app_resolves_the_step():
    cb, seen = _echo({"status": "connected", "connectors": []})

    setup_mcp_tool(server="notion", callback=cb)

    assert seen["action"] == "connect"


def test_merges_singular_and_list_forms_in_order_without_duplicates():
    cb, seen = _echo({"status": "connected", "connectors": []})

    setup_mcp_tool(server="atlassian", servers=["figma", "atlassian", "notion"], callback=cb)

    assert seen["servers"] == ["atlassian", "figma", "notion"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('["figma", "notion"]', ["figma", "notion"]),
        ("figma, notion", ["figma", "notion"]),
        ("figma", ["figma"]),
    ],
)
def test_tolerates_a_stringified_servers_list(raw, expected):
    """Models routinely send an array param as a JSON or comma string."""
    cb, seen = _echo({"status": "connected", "connectors": []})

    setup_mcp_tool(servers=raw, callback=cb)

    assert seen["servers"] == expected


def test_rejects_more_connectors_than_a_card_can_reasonably_ask():
    result = json.loads(
        setup_mcp_tool(
            servers=[f"server-{i}" for i in range(_MAX_CONNECTORS + 1)],
            callback=lambda *a: "",
        )
    )
    assert "error" in result


def test_multi_connector_breakdown_passes_through():
    answer = {
        "status": "partial",
        "connectors": [
            {"server": "figma", "status": "connected", "tools": ["get_file"]},
            {"server": "notion", "status": "declined"},
        ],
    }
    cb, _ = _echo(answer)

    result = json.loads(setup_mcp_tool(servers=["figma", "notion"], callback=cb))

    assert result["status"] == "partial"
    assert [c["server"] for c in result["connectors"]] == ["figma", "notion"]
    # Top-level `server` stays populated for anything still reading the scalar.
    assert result["server"] == "figma"


def test_timeout_returns_unanswered_not_error():
    result = json.loads(setup_mcp_tool(servers=["figma", "notion"], callback=lambda *a: ""))

    assert result["status"] == "unanswered"
    assert result["server"] == "figma"
    assert result["servers"] == ["figma", "notion"]


def test_callback_exception_is_tool_error():
    def cb(*a):
        raise RuntimeError("gateway went away")

    result = json.loads(setup_mcp_tool(server="figma", callback=cb))
    assert "error" in result


def test_non_json_answer_wrapped_as_error_status():
    result = json.loads(setup_mcp_tool(server="figma", callback=lambda *a: "garbage"))
    assert result["status"] == "error"


def test_steps_reach_a_callback_that_takes_them():
    seen = {}

    def callback(servers, action, reason, steps=()):
        seen["steps"] = list(steps)
        return json.dumps({"status": "declined", "server": servers[0]})

    setup_mcp_tool(
        server="acme",
        steps=["Create a token at [Settings](https://acme.test/settings)", "  ", "Paste it below"],
        callback=callback,
    )

    # Blanks dropped, order kept — a checklist with a hole in it reads as a
    # step the user missed.
    assert seen["steps"] == [
        "Create a token at [Settings](https://acme.test/settings)",
        "Paste it below",
    ]


def test_a_callback_that_predates_steps_still_gets_its_card():
    """The card matters more than the checklist. A surface wired before steps
    existed should show the connector, not raise a TypeError at the user."""
    calls = []

    def old_callback(servers, action, reason):
        calls.append(servers)
        return json.dumps({"status": "declined", "server": servers[0]})

    result = json.loads(setup_mcp_tool(server="acme", steps=["do a thing"], callback=old_callback))

    assert result["status"] == "declined"
    assert calls == [["acme"]]


def test_an_essay_is_trimmed_to_a_checklist():
    seen = {}

    def callback(servers, action, reason, steps=()):
        seen["steps"] = list(steps)
        return json.dumps({"status": "declined", "server": servers[0]})

    setup_mcp_tool(server="acme", steps=[f"step {i}" for i in range(20)], callback=callback)

    assert len(seen["steps"]) == 6


@pytest.mark.parametrize("action", ["connect", "install", "enable", "authorize"])
def test_all_actions_accepted(action):
    result = json.loads(
        setup_mcp_tool(
            server="x",
            action=action,
            callback=lambda s, a, r: json.dumps({"status": "declined", "server": s[0]}),
        )
    )
    assert result["status"] == "declined"
