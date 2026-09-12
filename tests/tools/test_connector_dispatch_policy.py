"""Real connector dispatch must run policies against each composed name."""

import json

import pytest


@pytest.mark.parametrize("blocked_by", ["hook", "execution"])
def test_remote_entries_run_request_hook_and_execution_policies(monkeypatch, blocked_by):
    import model_tools
    import hermes_cli.plugins as plugins
    from tools.registry import invalidate_check_fn_cache
    from tools.tool_gateway import bridge, config

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    denied = "connectors__gmail__SEND_EMAIL"
    rewritten = "connectors__slack__POST_MESSAGE"
    calls = [{"name": name, "arguments": {"body": "original"}} for name in (denied, rewritten)]
    events = []
    wire = []

    def request(**kw):
        events.append(("request", kw["tool_name"]))
        assert kw["args"] == {"body": "original"}
        assert kw["session_id"] == "policy-session"
        return {"args": {"body": "request-rewrite"}, "source": "test-policy"}

    def hook(name, args, **kw):
        events.append(("hook", name))
        assert args == {"body": "request-rewrite"}
        assert kw["middleware_trace"] == [{"source": "test-policy"}]
        assert kw["tool_call_id"] == "policy-call"
        if name == denied and blocked_by == "hook":
            return "hook denied", None
        return None, {"body": "hook-rewrite"}

    def execution(**kw):
        events.append(("execution", kw["tool_name"]))
        assert kw["args"] == {"body": "hook-rewrite"}
        assert kw["original_args"] == {"body": "original"}
        assert kw["session_id"] == "policy-session"
        if kw["tool_name"] == denied:
            return json.dumps({"error": {"code": "POLICY_DENIED", "message": "execution denied", "policy": "no-mail"}})
        return kw["next_call"]({})  # Empty dict must reach the wire, not original arguments.

    monkeypatch.setattr(plugins.get_plugin_manager(), "_middleware", {
        "tool_request": [request], "tool_execution": [execution]})
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", hook)

    class Client:
        def execute(self, planned):
            wire.extend(planned)
            return [{"data": "remote-ok", "error": None} for _ in planned]

    monkeypatch.setattr(bridge, "_default_client_factory", Client)
    kwargs = dict(enabled_toolsets=["connections"], session_id="policy-session", tool_call_id="policy-call",
                  skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
                  skip_tool_execution_middleware=True)
    result = json.loads(model_tools.handle_function_call("tool_call", {"calls": calls}, **kwargs))
    assert "denied" in json.dumps(result["results"][0]["error"])
    if blocked_by == "execution":
        assert result["results"][0]["error"] == {
            "code": "POLICY_DENIED", "message": "execution denied", "policy": "no-mail"}
    assert result["results"][1]["response"] == "remote-ok"
    assert [(p.name, p.arguments) for p in wire] == [(rewritten, {})]
    expected_denied = [("request", denied), ("hook", denied)]
    if blocked_by == "execution":
        expected_denied.append(("execution", denied))
    assert events == expected_denied + [(phase, rewritten) for phase in ("request", "hook", "execution")]
    assert result["total_count"] == 2 and result["success_count"] == result["error_count"] == 1

    wire.clear()
    result = json.loads(model_tools.handle_function_call("tool_call", {"calls": calls[:1]}, **kwargs))
    assert result["error_count"] == 1
    assert not wire  # An entirely blocked batch never constructs/sends an execute request.


def test_stop_during_a_connector_batch_leaves_unstarted_entries_unsent(monkeypatch):
    import model_tools
    from tools.interrupt import set_interrupt
    from tools.registry import invalidate_check_fn_cache
    from tools.tool_gateway import bridge, config

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    wire = []

    class Client:
        def execute(self, planned):
            wire.extend(planned)
            set_interrupt(True)  # /stop lands while the first entry is on the wire.
            return [{"data": "remote-ok", "error": None} for _ in planned]

    monkeypatch.setattr(bridge, "_default_client_factory", Client)
    calls = [{"name": f"connectors__gmail__{tool}", "arguments": {}}
             for tool in ("FETCH_EMAILS", "SEND_EMAIL", "CREATE_DRAFT")]
    try:
        result = json.loads(model_tools.handle_function_call(
            "tool_call", {"calls": calls}, enabled_toolsets=["connections"], session_id="stop-session",
            skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True))
    finally:
        set_interrupt(False)
    assert [p.name for p in wire] == [calls[0]["name"]]
    assert result["results"][0]["response"] == "remote-ok"
    assert [(e["index"], e["name"], e["error"]["code"]) for e in result["results"][1:]] == [
        (1, calls[1]["name"], "INTERRUPTED"), (2, calls[2]["name"], "INTERRUPTED")]
    assert result["total_count"] == 3 and result["success_count"] == 1 and result["error_count"] == 2


def test_disabled_connections_cannot_be_called_through_a_stale_schema(monkeypatch):
    from tools import connections_tool
    from tools.registry import registry

    monkeypatch.setattr(connections_tool, "_connectors_available", lambda: False)
    monkeypatch.setattr(connections_tool, "_default_client",
                        lambda: (_ for _ in ()).throw(AssertionError("disabled connector attempted I/O")))
    result = json.loads(registry.dispatch("manage_connections", {"action": "connect", "connectors": ["gmail"]}))
    assert "not available" in result["error"]
