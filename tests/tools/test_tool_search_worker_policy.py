"""Owned-worker disclosure uses native tool schemas and session identity."""
from tools.tool_search import ToolSearchConfig, assemble_tool_defs


TERMINALS = {
    "kanban_complete", "kanban_block", "kanban_schedule",
    "kanban_request_review", "kanban_request_changes",
}


def test_narrow_keeps_available_terminal_tools_direct(monkeypatch):
    from model_tools import get_tool_definitions

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_disclosure")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    raw = get_tool_definitions(["kanban"], skip_tool_search_assembly=True)
    available = {td["function"]["name"] for td in raw}
    assert TERMINALS <= available
    result = assemble_tool_defs(raw, config=ToolSearchConfig.from_raw({
        "enabled": "on", "kanban_worker_policy": "narrow", "defer": sorted(TERMINALS),
    }))
    direct = {td["function"]["name"] for td in result.tool_defs}
    assert TERMINALS <= direct


def test_narrow_catalog_and_bridge_agree_with_explicit_and_native_config(monkeypatch):
    import json

    from hermes_cli.config import save_config
    from model_tools import get_tool_definitions
    from tools import tool_search as ts

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_disclosure")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    raw = get_tool_definitions(["kanban"], skip_tool_search_assembly=True)
    names = {td["function"]["name"] for td in raw}
    cold = {
        "kanban_attach", "kanban_attach_url", "kanban_attachments", "kanban_comment",
        "kanban_heartbeat", "kanban_link", "kanban_unblock",
    } & names
    assert "kanban_comment" in cold
    cfg = ts.ToolSearchConfig.from_raw({"enabled": "on", "defer": [], "kanban_worker_policy": "narrow"})
    result = ts.assemble_tool_defs(raw, config=cfg)
    direct = {td["function"]["name"] for td in result.tool_defs}
    assert not cold & direct
    listing = next(td["function"]["description"] for td in result.tool_defs if td["function"]["name"] == "tool_search")
    assert all(name in listing for name in cold)
    # Explicit config must win even when the profile requests the opposite.
    save_config({"tools": {"tool_search": {"defer": [], "kanban_worker_policy": "off"}}})
    described = json.loads(ts.dispatch_tool_describe({"names": sorted(cold)}, current_tool_defs=raw, config=cfg))
    assert set(described["tools"]) == cold
    searched = json.loads(ts.dispatch_tool_search({"queries": ["kanban_comment"]}, current_tool_defs=raw, config=cfg))
    assert "kanban_comment" in searched["tools"]
    save_config({"tools": {"tool_search": {"defer": [], "kanban_worker_policy": "narrow"}}})
    assert cold <= ts.scoped_deferrable_names(raw)
    assert not TERMINALS & ts.scoped_deferrable_names(raw)
    args = {"task_id": "t_disclosure", "comment": "exact unchanged"}
    name, resolved, error = ts.resolve_underlying_call({"calls": [{"name": "kanban_comment", "arguments": args}]})
    assert (name, resolved, error) == ("kanban_comment", args, None)
    assert ts.resolve_underlying_call({"calls": [{"name": "kanban_schedule", "arguments": {}}]})[2]
    baseline = ts.ToolSearchConfig.from_raw({"defer": []})
    described = json.loads(ts.dispatch_tool_describe({"names": ["kanban_comment"]}, current_tool_defs=raw, config=baseline))
    assert "kanban_comment" not in described["tools"]


def test_registered_native_config_controls_runtime_and_disabled_scope(monkeypatch):
    import json

    from hermes_cli.config import load_config_readonly, save_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from model_tools import get_tool_definitions, handle_function_call
    from tools import tool_search as ts

    assert DEFAULT_CONFIG["tools"]["tool_search"].get("kanban_worker_policy") == "off"
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_native")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    save_config({"tools": {"tool_search": {"kanban_worker_policy": "narrow"}}})
    assert load_config_readonly()["tools"]["tool_search"]["kanban_worker_policy"] == "narrow"
    defs = get_tool_definitions(["kanban"], quiet_mode=True)
    names = {td["function"]["name"] for td in defs}
    assert TERMINALS <= names and "kanban_comment" not in names
    found = json.loads(handle_function_call("tool_describe", {"names": ["kanban_comment"]}, enabled_toolsets=["kanban"]))
    assert "kanban_comment" in found["tools"]
    denied = json.loads(handle_function_call("tool_call", {"calls": [{"name": "kanban_comment", "arguments": {}}]}, enabled_toolsets=["kanban"], disabled_toolsets=["kanban"]))
    assert "not available" in denied["error"]
    for invalid in [None, True, False, 1, [], {}, "unknown", " NARROW "]:
        assert ts.ToolSearchConfig.from_raw({"kanban_worker_policy": invalid}).kanban_worker_policy == "off"
    raw = get_tool_definitions(["kanban"], skip_tool_search_assembly=True)
    off = ts.assemble_tool_defs(raw, config=ts.ToolSearchConfig.from_raw({"enabled": "off", "kanban_worker_policy": "narrow"}))
    assert not off.activated and off.tool_defs == raw
