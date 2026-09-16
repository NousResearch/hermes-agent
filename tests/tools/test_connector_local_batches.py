"""Local deferred tools retain the live agent path; batches must not bypass it."""

import json
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mixed", [False, True])
def test_local_batches_rejected_before_any_entry_executes(monkeypatch, mixed):
    import model_tools
    from tools.tool_search import resolve_underlying_call
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    calls = [
        {"name": "session_search", "arguments": {}},
        {"name": "connectors__gmail__SEND_EMAIL" if mixed else "todo_list", "arguments": {}},
    ]
    name, args, error = resolve_underlying_call({"calls": calls})
    assert name is None and "one entry per tool_call" in error
    invoked = []
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **kw: invoked.append(a))
    monkeypatch.setattr(bridge, "_default_client_factory", lambda: invoked.append("gateway"))
    result = json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": calls}, enabled_toolsets=["connections", "session_search", "todo"]))
    assert "one entry per tool_call" in result["error"]
    assert invoked == []


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
                    {"targets": [{"name": t["name"], "status": "declined"} for t in payload["targets"]]}))
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
