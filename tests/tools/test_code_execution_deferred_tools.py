import json

import model_tools
from tools import code_execution_tool
from tools.registry import registry
from tools.tool_search import ToolSearchConfig


def _probe_schema(name: str) -> dict:
    return {
        "name": name,
        "description": "Deferred execute_code probe",
        "parameters": {"type": "object", "properties": {}},
    }


def test_local_execute_code_schema_advertises_deferred_bridges(monkeypatch):
    probe_name = "execute_code_deferred_schema_probe"
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_deferred_schema_probe",
        schema=_probe_schema(probe_name),
        handler=lambda _args, **_kwargs: json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    model_tools._clear_tool_defs_cache()

    try:
        definitions = model_tools.get_tool_definitions(
            enabled_toolsets=[
                "code_execution",
                "plugin_execute_code_deferred_schema_probe",
            ],
            quiet_mode=True,
        )
    finally:
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    execute_schema = next(
        item["function"]
        for item in definitions
        if item["function"]["name"] == "execute_code"
    )
    description = execute_schema["description"]
    assert "tool_search(queries:" in description
    assert "tool_describe(names:" in description
    assert "tool_call(name:" in description


def test_remote_execute_code_schema_hides_deferred_bridges(monkeypatch):
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "ssh"},
    )

    schema = code_execution_tool.build_execute_code_schema(
        code_execution_tool.SANDBOX_ALLOWED_TOOLS
        | code_execution_tool.DEFERRED_BRIDGE_TOOLS,
    )

    description = schema["description"]
    assert "tool_search(queries:" not in description
    assert "tool_describe(names:" not in description
    assert "tool_call(name:" not in description


def test_local_execute_code_calls_deferred_tool_in_session_scope(monkeypatch):
    probe_name = "execute_code_deferred_runtime_probe"
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_deferred_runtime_probe",
        schema={
            "name": probe_name,
            "description": "Deferred execute_code runtime probe",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {"value": args["value"], "session_id": kwargs.get("session_id")}
        ),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    model_tools._clear_tool_defs_cache()

    try:
        toolsets = [
            "code_execution",
            "plugin_execute_code_deferred_runtime_probe",
        ]
        model_tools.get_tool_definitions(
            enabled_toolsets=toolsets,
            quiet_mode=True,
        )
        raw = model_tools.handle_function_call(
            "execute_code",
            {
                "code": (
                    "from hermes_tools import tool_call\n"
                    f"result = tool_call({probe_name!r}, {{'value': 'nested'}})\n"
                    "print(result['value'])\n"
                    "print(result['session_id'])\n"
                )
            },
            task_id="execute-code-deferred-task",
            session_id="execute-code-deferred-session",
            enabled_toolsets=toolsets,
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    result = json.loads(raw)
    assert result["status"] == "success", raw
    assert result["output"].splitlines() == [
        "nested",
        "execute-code-deferred-session",
    ]


def test_registry_explicit_empty_scope_keeps_only_direct_helpers(monkeypatch):
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    resolved = code_execution_tool._sandbox_tools_for(
        [],
        allow_deferred_bridges=True,
    )
    schema = code_execution_tool.build_execute_code_schema(
        resolved,
        allow_deferred_bridges=True,
    )

    for name in code_execution_tool.DIRECT_SANDBOX_TOOLS:
        assert f"{name}(" in schema["description"]
    for name in code_execution_tool.DEFERRED_BRIDGE_TOOLS:
        assert f"{name}(" not in schema["description"]

    names = sorted(
        code_execution_tool.DIRECT_SANDBOX_TOOLS
        | code_execution_tool.DEFERRED_BRIDGE_TOOLS
    )
    code = (
        "import json, hermes_tools\n"
        f"names = {names!r}\n"
        "print(json.dumps({name: hasattr(hermes_tools, name) for name in names}))"
    )
    try:
        raw = registry.dispatch(
            "execute_code",
            {"code": code},
            task_id="execute-code-explicit-empty-scope",
            enabled_tools=[],
            enabled_toolsets=[],
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()

    result = json.loads(raw)
    assert result["status"] == "success", raw
    runtime_surface = json.loads(result["output"])
    for name in code_execution_tool.DIRECT_SANDBOX_TOOLS:
        assert runtime_surface[name]
    for name in code_execution_tool.DEFERRED_BRIDGE_TOOLS:
        assert not runtime_surface[name]


def test_execute_code_without_explicit_toolset_scope_hides_deferred_bridges(monkeypatch):
    probe_name = "execute_code_unscoped_deferred_probe"
    calls = []
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_unscoped_deferred_probe",
        schema=_probe_schema(probe_name),
        handler=lambda args, **kwargs: calls.append((args, kwargs)) or json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )

    try:
        raw = code_execution_tool.execute_code(
            (
                "from hermes_tools import tool_call\n"
                f"print(tool_call({probe_name!r}, {{}}))\n"
            ),
            task_id="execute-code-unscoped-task",
            enabled_tools=["tool_search", "tool_describe", "tool_call"],
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    result = json.loads(raw)
    assert result["status"] == "error", raw
    assert "cannot import name 'tool_call'" in result["error"]
    assert calls == []


def test_deferred_bridge_cannot_call_tool_outside_session_scope(monkeypatch):
    probe_name = "execute_code_out_of_scope_deferred_probe"
    calls = []
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_out_of_scope_deferred_probe",
        schema=_probe_schema(probe_name),
        handler=lambda args, **kwargs: calls.append((args, kwargs)) or json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )

    try:
        raw = code_execution_tool.execute_code(
            (
                "from hermes_tools import tool_call\n"
                f"result = tool_call({probe_name!r}, {{}})\n"
                "print(result['error'])\n"
            ),
            task_id="execute-code-scoped-task",
            enabled_tools=["tool_search", "tool_describe", "tool_call"],
            enabled_toolsets=["code_execution"],
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    result = json.loads(raw)
    assert result["status"] == "success", raw
    assert "not available" in result["output"]
    assert calls == []


def test_remote_rpc_dispatch_forwards_parent_session_scope(monkeypatch):
    from tools.code_execution_rpc import _default_dispatch

    captured = {}

    def fake_handle(tool_name, tool_args, **kwargs):
        captured.update(tool_name=tool_name, tool_args=tool_args, kwargs=kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools, "handle_function_call", fake_handle)

    dispatch = _default_dispatch(
        "remote-parent-task",
        session_id="remote-parent-session",
        enabled_toolsets=["code_execution", "web"],
        disabled_toolsets=["browser"],
    )
    raw = dispatch("tool_call", {"name": "probe", "arguments": {}})

    assert json.loads(raw) == {"ok": True}
    assert captured == {
        "tool_name": "tool_call",
        "tool_args": {"name": "probe", "arguments": {}},
        "kwargs": {
            "task_id": "remote-parent-task",
            "session_id": "remote-parent-session",
            "enabled_toolsets": ["code_execution", "web"],
            "disabled_toolsets": ["browser"],
        },
    }


def test_execute_code_forwards_parent_scope_to_remote_runner(monkeypatch):
    captured = {}

    def fake_remote(code, task_id, enabled_tools, **kwargs):
        captured.update(
            code=code,
            task_id=task_id,
            enabled_tools=enabled_tools,
            kwargs=kwargs,
        )
        return json.dumps({"status": "success", "output": ""})

    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "ssh"},
    )
    monkeypatch.setattr(code_execution_tool, "_execute_remote", fake_remote)

    raw = code_execution_tool.execute_code(
        "print('remote')",
        task_id="remote-task",
        session_id="remote-session",
        enabled_tools=["read_file"],
        enabled_toolsets=["code_execution", "file"],
        disabled_toolsets=["browser"],
        reset=True,
    )

    assert json.loads(raw)["status"] == "success"
    assert captured == {
        "code": "print('remote')",
        "task_id": "remote-task",
        "enabled_tools": ["read_file"],
        "kwargs": {
            "session_id": "remote-session",
            "enabled_toolsets": ["code_execution", "file"],
            "disabled_toolsets": ["browser"],
            "reset": True,
        },
    }
