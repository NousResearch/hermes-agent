import json

import pytest


@pytest.fixture(autouse=True)
def clear_tool_cache():
    import model_tools

    model_tools._clear_tool_defs_cache()
    yield
    model_tools._clear_tool_defs_cache()


def _loads(result):
    if isinstance(result, str):
        return json.loads(result)
    return result


def test_schema_returns_compact_exact_method_definition():
    from tools.hermes_codemode_tool import hermes_codemode_schema

    data = _loads(hermes_codemode_schema(methods=["read_file"]))

    assert data["schemaVersion"] == 1
    assert data["toolCount"] >= 1
    assert data["requestedMethods"] == ["read_file"]
    assert data["definitions"][0]["name"] == "read_file"
    assert data["definitions"][0]["mutating"] is False
    assert "inputSchema" in data["definitions"][0]
    field_names = {field["name"] for field in data["definitions"][0]["inputFields"]}
    assert {"path", "offset", "limit"}.issubset(field_names)


def test_schema_searches_live_tool_catalog_without_full_schemas():
    from tools.hermes_codemode_tool import hermes_codemode_schema

    data = _loads(hermes_codemode_schema(query="run shell command", limit=5))

    names = [match["name"] for match in data["matches"]]
    assert "terminal" in names
    assert data["definitions"] == []
    assert all("inputSchema" not in match for match in data["matches"])


def test_plan_execute_allows_read_only_calls_and_denies_mutations(monkeypatch):
    from tools import hermes_codemode_tool as cm

    calls = []

    def fake_handle_function_call(function_name, function_args, **kwargs):
        calls.append((function_name, function_args, kwargs))
        return json.dumps({"ok": True, "tool": function_name, "args": function_args})

    monkeypatch.setattr(cm, "_dispatch_tool_call", fake_handle_function_call)

    read_result = _loads(cm.hermes_codemode_execute(
        code="""
def main(codemode):
    return codemode.read_file({"path": "README.md", "limit": 3})
""",
        mode="plan",
        enabled_toolsets=["file"],
    ))

    assert read_result["ok"] is True
    assert read_result["result"]["tool"] == "read_file"
    assert calls[0][0] == "read_file"

    denied = _loads(cm.hermes_codemode_execute(
        code="""
def main(codemode):
    return codemode.write_file({"path": "x", "content": "y"})
""",
        mode="plan",
        enabled_toolsets=["file"],
    ))

    assert denied["ok"] is False
    assert "not allowed in plan mode" in denied["error"]


def test_execute_scopes_methods_to_enabled_toolsets(monkeypatch):
    from tools import hermes_codemode_tool as cm

    def fake_handle_function_call(function_name, function_args, **kwargs):
        return json.dumps({"ok": True})

    monkeypatch.setattr(cm, "_dispatch_tool_call", fake_handle_function_call)

    denied = _loads(cm.hermes_codemode_execute(
        code="""
def main(codemode):
    return codemode.terminal({"command": "pwd"})
""",
        mode="apply",
        enabled_toolsets=["file"],
    ))

    assert denied["ok"] is False
    assert "not available in this codemode session" in denied["error"]


def test_lean_codemode_cli_toolset_exposes_only_bridge_but_backs_full_cli_methods():
    from model_tools import get_tool_definitions
    from tools.hermes_codemode_tool import hermes_codemode_schema

    visible = get_tool_definitions(enabled_toolsets=["hermes-codemode-cli"], quiet_mode=True)
    visible_names = {tool["function"]["name"] for tool in visible}

    assert visible_names == {
        "hermes_codemode_status",
        "hermes_codemode_schema",
        "hermes_codemode_execute",
    }

    data = _loads(hermes_codemode_schema(
        query="run shell command",
        enabled_toolsets=["hermes-codemode-cli"],
    ))
    assert "terminal" in [match["name"] for match in data["matches"]]
