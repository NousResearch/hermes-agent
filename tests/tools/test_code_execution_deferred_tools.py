import base64
import json
import threading
from types import SimpleNamespace

import model_tools
import pytest
from tools import code_execution_rpc, code_execution_tool
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


def test_tool_definition_cache_tracks_effective_terminal_backend(monkeypatch):
    probe_name = "execute_code_backend_cache_probe"
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_backend_cache_probe",
        schema=_probe_schema(probe_name),
        handler=lambda _args, **_kwargs: json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "on"}),
    )
    terminal = {"env_type": "local"}
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: dict(terminal),
    )
    model_tools._clear_tool_defs_cache()

    def execute_code_description():
        definitions = model_tools.get_tool_definitions(quiet_mode=True)
        return next(
            item["function"]["description"]
            for item in definitions
            if item["function"]["name"] == "execute_code"
        )

    try:
        execute_code_description()  # settle lazy plugin discovery before warming the cache
        local_description = execute_code_description()
        terminal["env_type"] = "ssh"
        remote_description = execute_code_description()
    finally:
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    assert "tool_call(name:" in local_description
    assert "tool_call(name:" not in remote_description


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


@pytest.mark.parametrize("dispatch_path", ["sequential", "concurrent"])
def test_default_agent_can_call_advertised_deferred_bridge(monkeypatch, dispatch_path):
    from agent.agent_runtime_helpers import invoke_tool
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    probe_name = "execute_code_default_agent_deferred_probe"
    registry.register(
        name=probe_name,
        toolset="plugin_execute_code_default_agent_deferred_probe",
        schema=_probe_schema(probe_name),
        handler=lambda _args, **_kwargs: json.dumps({"value": "default-scope"}),
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

    definitions = model_tools.get_tool_definitions(quiet_mode=True)
    valid_names = {item["function"]["name"] for item in definitions}
    assert code_execution_tool.DEFERRED_BRIDGE_TOOLS <= valid_names
    agent = SimpleNamespace(
        _context_engine_tool_names=set(),
        _memory_manager=None,
        _current_turn_id="turn-default",
        _current_api_request_id="request-default",
        _should_emit_quiet_tool_messages=lambda: False,
        disabled_toolsets=None,
        enabled_toolsets=None,
        quiet_mode=True,
        session_id="execute-code-default-agent-session",
        valid_tool_names=valid_names,
    )
    code = (
        "from hermes_tools import tool_call\n"
        f"print(tool_call({probe_name!r}, {{}})['value'])\n"
    )
    ref = _ToolCallRef(
        name="execute_code",
        args={"code": code},
        task_id="execute-code-default-agent-task",
        call_id="call-default",
        trace=[],
    )

    try:
        if dispatch_path == "sequential":
            raw = _resolve_sequential_dispatch(agent, ref, []).execute(ref.args)
        else:
            raw = invoke_tool(
                agent,
                ref.name,
                ref.args,
                ref.task_id,
                ref.call_id,
                messages=[],
                pre_tool_block_checked=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    result = json.loads(raw)
    assert result["status"] == "success", raw
    assert result["output"].strip() == "default-scope"


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


def test_late_local_rpc_request_cannot_use_next_cell_session(monkeypatch):
    probe_name = "execute_code_late_cell_probe"
    calls = []
    toolset = "plugin_execute_code_late_cell_probe"
    registry.register(
        name=probe_name,
        toolset=toolset,
        schema=_probe_schema(probe_name),
        handler=lambda _args, **kwargs: calls.append(kwargs.get("session_id"))
        or json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    common = {
        "task_id": "execute-code-late-cell-task",
        "enabled_tools": ["tool_call"],
        "enabled_toolsets": [toolset],
    }

    try:
        first = code_execution_tool.execute_code(
            (
                "import threading\n"
                "from hermes_tools import tool_call\n"
                "late_call_gate = threading.Event()\n"
                "late_call_done = threading.Event()\n"
                "def late_call():\n"
                "    late_call_gate.wait()\n"
                f"    tool_call({probe_name!r}, {{}})\n"
                "    late_call_done.set()\n"
                "threading.Thread(target=late_call, daemon=True).start()\n"
            ),
            session_id="cell-session-a",
            **common,
        )
        second = code_execution_tool.execute_code(
            "late_call_gate.set()\nassert late_call_done.wait(2)\nprint('settled')\n",
            session_id="cell-session-b",
            **common,
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    assert json.loads(first)["status"] == "success", first
    assert json.loads(second)["status"] == "success", second
    assert calls == []


def test_reused_thread_pool_captures_each_cells_authority(monkeypatch):
    probe_name = "execute_code_thread_pool_cell_probe"
    calls = []
    toolset = "plugin_execute_code_thread_pool_cell_probe"
    registry.register(
        name=probe_name,
        toolset=toolset,
        schema=_probe_schema(probe_name),
        handler=lambda _args, **kwargs: calls.append(kwargs.get("session_id"))
        or json.dumps({"ok": True}),
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    common = {
        "task_id": "execute-code-thread-pool-cell-task",
        "enabled_tools": ["tool_call"],
        "enabled_toolsets": [toolset],
    }

    try:
        first = code_execution_tool.execute_code(
            (
                "from concurrent.futures import ThreadPoolExecutor\n"
                "from hermes_tools import tool_call\n"
                "pool = ThreadPoolExecutor(max_workers=1)\n"
                f"print(pool.submit(tool_call, {probe_name!r}, {{}}).result())\n"
            ),
            session_id="cell-session-a",
            **common,
        )
        second = code_execution_tool.execute_code(
            f"print(pool.submit(tool_call, {probe_name!r}, {{}}).result())\n",
            session_id="cell-session-b",
            **common,
        )
    finally:
        from tools.code_kernel import _REGISTRY

        _REGISTRY.shutdown()
        registry.deregister(probe_name)
        model_tools._clear_tool_defs_cache()

    assert json.loads(first)["status"] == "success", first
    assert json.loads(second)["status"] == "success", second
    assert calls == ["cell-session-a", "cell-session-b"]


def test_local_rpc_serializes_structured_result_as_one_json_frame():
    result = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "first line\nsecond line"}],
    }
    request = json.dumps(
        {"token": "rpc-token", "tool": "tool_call", "args": {}}
    ).encode() + b"\n"

    class Connection:
        def __init__(self):
            self.reads = [request, b""]
            self.sent = bytearray()

        def settimeout(self, _timeout):
            return None

        def recv(self, _size):
            return self.reads.pop(0)

        def sendall(self, payload):
            self.sent.extend(payload)

        def close(self):
            return None

    connection = Connection()

    class Server:
        def settimeout(self, _timeout):
            return None

        def accept(self):
            return connection, None

    code_execution_rpc._rpc_server_loop(
        Server(),
        "task-rpc",
        [],
        [0],
        1,
        frozenset({"tool_call"}),
        threading.Event(),
        "rpc-token",
        dispatch=lambda _name, _args: result,
    )

    assert connection.sent.count(b"\n") == 1
    assert json.loads(connection.sent) == result


def test_local_rpc_binds_cell_authority_before_dispatch():
    request = json.dumps(
        {
            "token": "rpc-token",
            "cell_token": "cell-a",
            "tool": "tool_call",
            "args": {},
        }
    ).encode() + b"\n"

    class Connection:
        def __init__(self):
            self.reads = [request, b""]
            self.sent = bytearray()

        def settimeout(self, _timeout):
            return None

        def recv(self, _size):
            return self.reads.pop(0)

        def sendall(self, payload):
            self.sent.extend(payload)

        def close(self):
            return None

    connection = Connection()

    class Server:
        def settimeout(self, _timeout):
            return None

        def accept(self):
            return connection, None

    kernel = SimpleNamespace(cell_authority=None)

    class Authority:
        def __init__(self, owner, token, *, replace_with=None):
            self.owner = owner
            self._token = token
            self.replace_with = replace_with
            self.active = True

        @property
        def rpc_cell_token(self):
            if self.replace_with is not None:
                kernel.cell_authority = self.replace_with
            return self._token

        def dispatch(self, _name, _args):
            return {"owner": self.owner}

    authority_b = Authority("B", "cell-b")
    authority_a = Authority("A", "cell-a", replace_with=authority_b)
    kernel.cell_authority = authority_a

    from tools.code_kernel import _bind_cell_dispatch

    code_execution_rpc._rpc_server_loop(
        Server(),
        "task-rpc",
        [],
        [0],
        1,
        frozenset({"tool_call"}),
        threading.Event(),
        "rpc-token",
        bind_dispatch=lambda token: _bind_cell_dispatch(kernel, token),
    )

    assert json.loads(connection.sent) == {"owner": "A"}


def test_remote_rpc_serializes_structured_result(monkeypatch):
    result = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "remote result"}],
    }
    request = {
        "token": "rpc-token",
        "tool": "tool_call",
        "args": {},
        "seq": 7,
    }
    stop_event = threading.Event()

    class Environment:
        response = None

        def execute(self, command, **_kwargs):
            if command.startswith("ls -1"):
                return {"output": "/rpc/req_000007\n"}
            if command.startswith("cat "):
                stop_event.set()
                return {"output": json.dumps(request)}
            if command.startswith("echo '"):
                encoded = command.split("'", 2)[1]
                self.response = base64.b64decode(encoded).decode()
            return {"output": ""}

    environment = Environment()
    monkeypatch.setattr(
        model_tools,
        "handle_function_call",
        lambda _name, _args, **_kwargs: result,
    )

    code_execution_rpc._rpc_poll_loop(
        environment,
        "/rpc",
        "task-rpc",
        [],
        [0],
        1,
        frozenset({"tool_call"}),
        stop_event,
        "rpc-token",
    )

    assert json.loads(environment.response) == result


def test_remote_rpc_replies_when_cell_authority_is_stale():
    request = {
        "token": "rpc-token",
        "cell_token": "cell-a",
        "tool": "tool_call",
        "args": {},
        "seq": 7,
    }
    stop_event = threading.Event()

    class Environment:
        response = None

        def execute(self, command, **_kwargs):
            if command.startswith("ls -1"):
                return {"output": "/rpc/req_000007\n"}
            if command.startswith("cat "):
                stop_event.set()
                return {"output": json.dumps(request)}
            if command.startswith("echo '"):
                encoded = command.split("'", 2)[1]
                self.response = base64.b64decode(encoded).decode()
            return {"output": ""}

    environment = Environment()
    code_execution_rpc._rpc_poll_loop(
        environment,
        "/rpc",
        "task-rpc",
        [],
        [0],
        1,
        frozenset({"tool_call"}),
        stop_event,
        "rpc-token",
        cell_token="cell-b",
    )

    response = json.loads(environment.response)
    assert "expired" in response["error"].lower()


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
