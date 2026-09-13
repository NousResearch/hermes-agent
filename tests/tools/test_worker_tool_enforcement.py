"""Worker tool ceilings at the real registry and deferred-call boundaries."""

import json
from types import SimpleNamespace

from agent.delegation_model_routing import ToolPolicy
from tools.delegate_tool_toolsets import _apply_exact_tool_policy, _resolve_child_toolsets


def _schema(name):
    return {
        "name": name,
        "description": "Synthetic worker tool.",
        "parameters": {"type": "object", "properties": {}},
    }


def _tool(name):
    return {"type": "function", "function": _schema(name)}


def test_model_dispatch_forwards_worker_limit_to_execute_code(monkeypatch):
    """The ordinary model_tools signature reaches execute_code's nested RPC scope."""
    import model_tools
    from tools import code_execution_tool

    captured = {}

    def fake_execute_code(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(code_execution_tool, "execute_code", fake_execute_code)
    result = model_tools.handle_function_call(
        "execute_code",
        {"code": "print('bounded')"},
        task_id="worker-task",
        session_id="worker-session",
        enabled_tools=["read_file"],
        worker_max_tool_calls=2,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result) == {"ok": True}
    assert captured["enabled_tools"] == ["read_file"]
    assert captured["worker_max_tool_calls"] == 2


def test_model_dispatch_accepts_worker_limit_for_ordinary_registry_tool():
    """R1's original call contract no longer raises before a native handler."""
    import model_tools
    from tools.registry import registry

    name = "pytest_worker_native_read"
    registry.register(
        name=name,
        toolset="pytest-worker",
        schema=_schema(name),
        handler=lambda _args, **_kwargs: json.dumps({"called": True}),
    )
    try:
        result = model_tools.handle_function_call(
            name,
            {},
            worker_max_tool_calls=1,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
        assert json.loads(result) == {"called": True}
    finally:
        registry.deregister(name)


def test_exact_policy_intersects_ancestor_profile_and_empty_request():
    child = SimpleNamespace(
        valid_tool_names={"read_file", "write_file", "execute_code", "delegate_task"},
        tools=[_tool(name) for name in ("read_file", "write_file", "execute_code", "delegate_task")],
    )
    _apply_exact_tool_policy(
        child,
        ToolPolicy(allowed_tools=("read_file", "write_file")),
        request_blocked_tools=["write_file"],
        ancestor_allowed_tools={"read_file", "write_file", "delegate_task"},
    )
    assert child.valid_tool_names == {"read_file"}

    empty = SimpleNamespace(
        valid_tool_names={"read_file", "delegate_task"},
        tools=[_tool("read_file"), _tool("delegate_task")],
    )
    _apply_exact_tool_policy(
        empty,
        ToolPolicy(),
        request_toolsets=[],
        ancestor_allowed_tools={"read_file", "delegate_task"},
    )
    assert empty.valid_tool_names == set()
    assert empty._worker_effective_tool_names == frozenset()
    assert empty.tools == []


def test_explicit_and_ancestor_delegation_denials_override_worker_role(monkeypatch):
    parent = SimpleNamespace(enabled_toolsets=["file", "delegation"], disabled_toolsets=[])
    enabled, _disabled = _resolve_child_toolsets(parent, ["file"], "orchestrator")
    assert "delegation" not in enabled

    parent.disabled_toolsets = ["delegation"]
    enabled, disabled = _resolve_child_toolsets(parent, None, "orchestrator")
    assert "delegation" in disabled

    child = SimpleNamespace(
        valid_tool_names={"read_file", "delegate_task"},
        tools=[_tool("read_file"), _tool("delegate_task")],
    )
    _apply_exact_tool_policy(child, ToolPolicy(allowed_toolsets=("file",)))
    assert "delegate_task" not in child._worker_effective_tool_names


def test_deferred_mcp_call_cannot_bypass_exact_worker_names():
    from agent.tool_executor import _parse_tool_call
    from tools.registry import registry

    name = "mcp__pytest_worker__private_read"
    registry.register(
        name=name,
        toolset="mcp-pytest-worker",
        schema=_schema(name),
        handler=lambda _args, **_kwargs: json.dumps({"leaked": True}),
    )
    try:
        agent = SimpleNamespace(
            enabled_toolsets=["mcp-pytest-worker"],
            disabled_toolsets=[],
            valid_tool_names={"read_file"},
            _worker_effective_tool_names=frozenset({"read_file"}),
        )
        call = SimpleNamespace(
            id="mcp-denied",
            function=SimpleNamespace(
                name="tool_call",
                arguments=json.dumps({"name": name, "arguments": {}}),
            ),
        )
        parsed = _parse_tool_call(agent, call)
        assert parsed.name == "tool_call"
        assert "not available in this session" in parsed.scope_block
    finally:
        registry.deregister(name)


def test_deferred_catalog_dispatch_is_independent_of_visible_schema(monkeypatch):
    import model_tools
    from agent.tool_executor import _parse_tool_call
    from tools.registry import registry

    name = "mcp__pytest_worker__catalog_read"
    registry.register(
        name=name,
        toolset="mcp-pytest-worker",
        schema=_schema(name),
        handler=lambda _args, **_kwargs: json.dumps({"called": name}),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config_readonly",
        lambda: SimpleNamespace(effective_defer_tools=frozenset({name})),
    )
    call = SimpleNamespace(
        id="mcp-allowed",
        function=SimpleNamespace(
            name="tool_call",
            arguments=json.dumps({"name": name, "arguments": {}}),
        ),
    )
    try:
        for agent in (
            SimpleNamespace(
                enabled_toolsets=["mcp-pytest-worker"], disabled_toolsets=[],
                valid_tool_names={"tool_call"}, _executable_tool_names={name},
            ),
            SimpleNamespace(
                enabled_toolsets=["mcp-pytest-worker"], disabled_toolsets=[],
                valid_tool_names={"tool_call"}, _worker_effective_tool_names=frozenset({name}),
            ),
        ):
            parsed = _parse_tool_call(agent, call)
            assert parsed.scope_block is None and parsed.name == name
            result = model_tools.handle_function_call(
                parsed.name, parsed.args,
                enabled_toolsets=agent.enabled_toolsets,
                disabled_toolsets=agent.disabled_toolsets,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )
            assert json.loads(result) == {"called": name}
    finally:
        registry.deregister(name)


def test_worker_native_and_execute_code_use_exact_execution_authority(monkeypatch):
    from agent.tool_executor import _ToolCallRef, _parse_tool_call, _resolve_sequential_dispatch
    from tools import code_execution_tool

    agent = SimpleNamespace(
        enabled_toolsets=["code_execution", "file"], disabled_toolsets=[],
        valid_tool_names={"execute_code"},
        _worker_effective_tool_names=frozenset({"execute_code", "read_file"}),
        _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False,
        session_id="worker-exact", _current_turn_id="", _current_api_request_id="",
    )
    denied = SimpleNamespace(
        id="native-denied",
        function=SimpleNamespace(name="write_file", arguments=json.dumps({"path": "x", "content": "y"})),
    )
    assert "not permitted" in _parse_tool_call(agent, denied).scope_block

    captured = {}
    monkeypatch.setattr(
        code_execution_tool, "execute_code",
        lambda **kwargs: captured.update(kwargs) or json.dumps({"ok": True}),
    )
    dispatch = _resolve_sequential_dispatch(
        agent,
        _ToolCallRef("execute_code", {"code": "print('bounded')"}, "task", "call", []),
        [],
    )
    assert json.loads(dispatch.execute({"code": "print('bounded')"})) == {"ok": True}
    assert set(captured["enabled_tools"]) == {"execute_code", "read_file"}
