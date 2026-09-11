"""Connector capabilities follow session grants, not process-wide credentials."""

import json

import pytest


@pytest.mark.parametrize("enabled,disabled,allowed", [
    ([], [], False),
    (["safe"], [], False),
    (["hermes-webhook"], [], False),
    (["connections"], [], True),
    (["hermes-cli"], ["connections"], False),
    (["safe"], ["connections"], False),
    (None, ["connections"], False),
    (None, [], True),
])
def test_connector_scope_controls_schema_discovery_and_execution(monkeypatch, enabled, disabled, allowed):
    import model_tools
    from tools.tool_gateway import bridge, config
    from tools import connections_tool

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    from tools.registry import invalidate_check_fn_cache
    invalidate_check_fn_cache()
    remote = []
    name = "connectors__gmail__SEND_EMAIL"

    class Client:
        def search(self, queries):
            remote.append("search")
            return {"results": [{"tools": ["GMAIL_SEND_EMAIL"]}], "schemas": {
                "GMAIL_SEND_EMAIL": {"connector": "gmail", "description": "Send mail", "input_schema": {}}}}

        def schemas(self, names):
            remote.append("describe")
            return {"schemas": {"GMAIL_SEND_EMAIL": {"description": "Send mail", "input_schema": {}}}}

        def execute(self, planned):
            remote.append("execute")
            return [{"data": "sent", "error": None} for _ in planned]

        def list_connectors(self):
            remote.append("status")
            return []

    monkeypatch.setattr(bridge, "_default_client_factory", Client)
    monkeypatch.setattr(connections_tool, "_default_client", Client)
    scope = {"enabled_toolsets": enabled, "disabled_toolsets": disabled}
    defs = model_tools.get_tool_definitions(**scope, quiet_mode=True, skip_tool_search_assembly=True)
    assert ("manage_connections" in {td["function"]["name"] for td in defs}) is allowed
    if enabled == []:
        assert model_tools.get_tool_definitions(**scope, quiet_mode=True) == []

    def call(tool, args):
        return json.loads(model_tools.handle_function_call(tool, args, **scope))

    assert (name in call("tool_search", {"queries": ["send mail"]})["tools"]) is allowed
    assert (name in call("tool_describe", {"names": [name]})["tools"]) is allowed
    result = call("tool_call", {"calls": [{"name": name, "arguments": {}}]})
    direct = call(name, {})
    status = call("manage_connections", {"action": "status"})
    if allowed:
        assert result["results"][0]["response"] == "sent"
        assert "error" not in status
        assert direct["response"] == "sent"
        assert remote == ["search", "describe", "execute", "execute", "status"]
    else:
        assert "not available in this session" in json.dumps(result)
        assert "not available in this session" in json.dumps(status)
        assert "not available in this session" in json.dumps(direct)
        assert remote == []


def test_ordinary_platform_defaults_grant_connections_without_widening_webhook():
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset

    for platform in ("cli", "telegram"):
        enabled = _get_platform_tools({}, platform)
        assert "connections" in enabled
        assert "manage_connections" in {name for ts in enabled for name in resolve_toolset(ts)}
    assert "connections" not in _get_platform_tools({}, "webhook")
    for selection in ([], ["safe"], ["file"]):
        assert "connections" not in _get_platform_tools({"platform_toolsets": {"cli": selection}}, "cli")
