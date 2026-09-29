"""Weak-model MCP calls resolve onto the registered tool and a batch runs once per call."""

import json

from tools.registry import registry
from tools.tool_search import resolve_underlying_call


def _register(name, description, properties, required=None):
    registry.register(
        name=name, toolset="mcp-notifuse", handler=lambda a, **kw: "{}",
        schema={"name": name, "description": description, "parameters": {
            "type": "object", "properties": properties, "required": required or [],
        }},
    )


def _drop(*names):
    for name in names:
        registry.deregister(name)


def test_chatwoot_call_without_an_account_id_uses_the_brand_id(monkeypatch):
    from tools.mcp_call_resolve import apply_inbox_account_id

    tool = "mcp__ivx-foundrly-chatwoot__chatwoot_list_conversations"
    _register(tool, "List conversations", {"accountId": {"type": "number"}, "status": {"type": "string"}})
    monkeypatch.setattr("tools.mcp_call_resolve._stored_inbox_account_id", lambda: "233")
    try:
        name, args, err = resolve_underlying_call({"name": tool, "arguments": {"status": "open"}})
        assert err is None
        assert name == tool
        assert args["accountId"] == 233
        assert args["status"] == "open"
        kept = {}
        apply_inbox_account_id(tool, kept, {"accountId": {"type": "number"}}, "999")
        kept["accountId"] = 10
        apply_inbox_account_id(tool, kept, {"accountId": {"type": "number"}}, "999")
        assert kept["accountId"] == 10
        other = {"workspace_id": "foundrly"}
        apply_inbox_account_id(
            "mcp__notifuse__notifuse_messages_list", other, {"workspace_id": {"type": "string"}, "accountId": {"type": "number"}}, "233",
        )
        assert "accountId" not in other
    finally:
        _drop(tool)


def test_click_guess_resolves_to_the_read_tool_and_keeps_the_id():
    list_name = "mcp__notifuse__notifuse_messages_list"
    send_name = "mcp__notifuse__notifuse_send_message"
    _register(list_name, "List broadcast messages", {
        "workspace_id": {"type": "string"},
        "broadcast_id": {"type": "string"},
        "has_clicked": {"type": "boolean"},
    })
    _register(send_name, "Send a broadcast to clicked recipients", {
        "workspace_id": {"type": "string"},
    })
    try:
        name, args, err = resolve_underlying_call({
            "name": "mcp__notifuse__get_recipient_clicks",
            "arguments": {"campaign_id": "camp-1", "note": "ignore"},
        })
        assert err is None
        assert name == list_name
        assert args["broadcast_id"] == "camp-1"
        assert args["has_clicked"] is True
        assert "campaign_id" not in args and "note" not in args
    finally:
        _drop(list_name, send_name)


def test_write_tool_is_not_used_for_a_read_guess():
    send_name = "mcp__notifuse__notifuse_send_message"
    _register(send_name, "Send the clicked campaign", {"workspace_id": {"type": "string"}})
    try:
        name, _args, err = resolve_underlying_call({
            "name": "mcp__notifuse__get_recipient_clicks", "arguments": {},
        })
        assert name is None
        assert "not a known tool name" in err
        assert send_name in err
    finally:
        _drop(send_name)


def test_mcp_batch_runs_each_distinct_call_once(monkeypatch):
    from model_tools import _CallIds
    from tools.mcp_call_resolve import MCP_BATCH_SENTINEL, dispatch_mcp_batch
    import model_tools

    list_name = "mcp__notifuse__notifuse_broadcasts_list"
    stats_name = "mcp__notifuse__notifuse_messages_broadcast_stats"
    _register(list_name, "List broadcasts", {"workspace_id": {"type": "string"}})
    _register(stats_name, "Broadcast stats", {"workspace_id": {"type": "string"}, "broadcast_id": {"type": "string"}})
    invoked = []

    def fake(name, args, **_kwargs):
        invoked.append(name)
        return json.dumps({"tool": name, "args": args})

    monkeypatch.setattr(model_tools, "handle_function_call", fake)
    try:
        sentinel, payload, err = resolve_underlying_call({"calls": [
            {"name": list_name, "arguments": {"workspace_id": "foundrly"}},
            {"name": list_name, "arguments": {"workspace_id": "foundrly"}},
            {"name": stats_name, "arguments": {"workspace_id": "foundrly", "broadcast_id": "b1"}},
        ]})
        assert err is None and sentinel == MCP_BATCH_SENTINEL
        body = json.loads(dispatch_mcp_batch(
            payload["calls"], _CallIds(), user_task=None, enabled_tools=None, middleware_trace=[],
            enabled_toolsets=None, disabled_toolsets=None,
            scoped_names=frozenset({list_name, stats_name}),
        ))
        assert invoked == [list_name, stats_name]
        assert [row["name"] for row in body["results"]] == [list_name, stats_name]
    finally:
        _drop(list_name, stats_name)
