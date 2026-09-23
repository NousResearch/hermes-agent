"""Local deferred tools retain the live agent path; batches must not bypass it."""

import json
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mixed", [False, True])
def test_local_batches_execute_every_entry_in_order(monkeypatch, mixed):
    import model_tools
    from tools.tool_search import resolve_underlying_call
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    local_names = ["mcp__batch__first", "mcp__batch__second"]
    calls = [{"name": local_names[0], "arguments": {"value": 1}}, {
        "name": "connectors__gmail__SEND_EMAIL" if mixed else local_names[1],
        "arguments": {"value": 2},
    }]
    name, args, error = resolve_underlying_call({"calls": calls})
    assert error is None and args["calls"] == calls
    invoked = []
    schemas = [
        {"type": "function", "function": {"name": local_name, "description": "batch test",
         "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}}}}
        for local_name in local_names
    ]
    if mixed:
        schemas.append({"type": "function", "function": {"name": "manage_connections", "parameters": {}}})
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: schemas)
    monkeypatch.setattr(model_tools, "_select_tool_names", lambda *a, **kw: {
        "manage_connections", *local_names})
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda tool_name, args, **kw: (
        invoked.append((tool_name, args)) or json.dumps({"tool": tool_name, "value": args.get("value")})))
    monkeypatch.setattr("tools.connectors.dispatch.dispatch_connector_call", lambda name, arguments, tool_call_id: (
        invoked.append((name, arguments)) or json.dumps({"response": {"tool": name, "value": arguments.get("value")}})))
    result = json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": calls}, enabled_toolsets=["connections"]))
    assert [entry["name"] for entry in result["results"]] == [call["name"] for call in calls]
    assert [entry["response"]["value"] for entry in result["results"]] == [1, 2]
    assert invoked == [(call["name"], call["arguments"]) for call in calls]


@pytest.mark.parametrize("flatten_probe", [False, True])
def test_single_local_unwrap_keeps_session_db_todo_store_and_setup_callback(tmp_path, flatten_probe):
    from agent.tool_executor import _unwrap_tool_search_call
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_state import SessionDB
    from tools.connectors import live
    from tools.connectors.contract import SettleReason
    from tools.connectors.mcp import apply_answer
    from tools.todo_tool import TodoStore

    from gateway.session_context import reset_session_vars, set_session_vars

    # A desktop session: the MCP card exists only there.
    set_session_vars(source="desktop", session_key="current-session", session_id="current-session")

    db = SessionDB(tmp_path / "recall.db")
    db.create_session("past-session", source="cli")
    db.append_message("past-session", role="user", content="live-db-proof")
    callbacks = []
    def connection(payload):
        callbacks.append(payload)

        def respond():
            operation = live.get("current-session", payload["op_id"])
            if operation is not None:
                apply_answer(operation, json.dumps(
                    {"targets": [{"name": t["name"], "status": "skipped"} for t in payload["targets"]]}))
                operation.settle(SettleReason.all_resolved)

        threading.Timer(0.02, respond).start()
        return None

    agent = SimpleNamespace(
        enabled_toolsets=["todo", "session_search", "connections"], disabled_toolsets=[],
        session_id="current-session", _todo_store=TodoStore(), _memory_manager=None,
        _get_session_db_for_recall=lambda: db, connection_callback=connection,
    )
    calls = [
        {"name": "session_search", "arguments": {"session_id": "past-session"}},
        {"name": "todo_list", "arguments": {"todos": [{"id": "a", "content": "live-store-proof", "status": "pending"}]}},
        {"name": "manage_connections", "arguments": {
            "action": "install", "connectors": [{"name": "linear", "mcp": True}]}},
    ]
    results = []
    try:
        for entry in calls:
            if entry["name"] == "manage_connections":
                # Not deferrable, so it reaches invoke_tool directly and must find the agent callback.
                name, args, error = entry["name"], entry["arguments"], None
            else:
                name, args, error = _unwrap_tool_search_call(
                    agent, "tool_call", {"calls": [entry]}, flatten_probe=flatten_probe)
            assert name == entry["name"] and error is None
            results.append(json.loads(invoke_tool(
                agent, name, args, "task", tool_call_id="call", pre_tool_block_checked=True)))
        assert "live-db-proof" in json.dumps(results[0])
        assert agent._todo_store.read()[0]["content"] == "live-store-proof"
        assert results[2]["targets"][0] == {
            "name": "linear", "kind": "mcp", "action": "install", "state": "skipped"}
        assert [(c["tool_call_id"], [t["name"] for t in c["targets"]]) for c in callbacks] == [("call", ["linear"])]
    finally:
        db.close()
        reset_session_vars()


def test_ordered_batch_reports_one_block_without_stopping_siblings(monkeypatch):
    import model_tools
    from tools.connectors.dispatch import dispatch_connector_batch

    invoked = []
    def dispatch(name, arguments, **kwargs):
        entry_name = arguments["calls"][0]["name"] if name == "tool_call" else name
        invoked.append(entry_name)
        return json.dumps({"error": "denied"} if entry_name == "mcp__batch__first" else {"ok": entry_name})

    monkeypatch.setattr(model_tools, "handle_function_call", dispatch)
    calls = [{"name": "mcp__batch__first", "arguments": {}},
             {"name": "mcp__batch__second", "arguments": {}}]
    result = json.loads(dispatch_connector_batch(
        calls, model_tools._CallIds(), user_task=None, enabled_tools=None,
        middleware_trace=[], enabled_toolsets=None, disabled_toolsets=None))
    assert invoked == ["mcp__batch__first", "mcp__batch__second"]
    assert result["results"][0]["error"]["message"] == "denied"
    assert result["results"][1]["response"] == {"ok": "mcp__batch__second"}
