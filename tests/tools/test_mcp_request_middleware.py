"""``mcp_request`` middleware sets a ``tools/call`` request's ``_meta``: JSON-RPC request metadata the
server reads and the model never sees. Real stdio child, real plugin discovery, real SDK round trip."""

import json
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SERVER = textwrap.dedent("""\
    import json
    from mcp.server.mcpserver import MCPServer, Context
    app = MCPServer("meta")

    @app.tool()
    async def whoami(ctx: Context) -> str:
        await ctx.report_progress(1, 1, "seen")
        return json.dumps(dict(ctx.request_context.meta or {}), default=str)

    app.run()
""")

PLUGIN = textwrap.dedent("""\
    def register(ctx):
        def tenant(meta, server_name, tool_name, session_id, args, **_):
            return {"meta": {**meta, "com.example/session": session_id, "com.example/call": f"{server_name}/{tool_name}",
                             "com.example/args-seen": sorted(args),
                             # Owned by the SDK and the spec, never by a plugin:
                             "progressToken": "forged", "modelcontextprotocol.io/forged": "forged"}}
        ctx.register_middleware("mcp_request", tenant)
""")


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_mcp_request_middleware_meta_reaches_the_server(monkeypatch, tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    home = tmp_path / "home"
    plugin = home / "plugins" / "tenant_meta"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: tenant_meta\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text(PLUGIN)
    (home / "server.py").write_text(SERVER, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [tenant_meta]\n"
        f"mcp_servers:\n  meta:\n    command: {json.dumps(sys.executable)}\n"
        f"    args: [{json.dumps(str(home / 'server.py'))}]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.tool_progress_relay import bound_tool_progress
    from hermes_cli import plugins
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.registry import registry

    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    try:
        assert "mcp__meta__whoami" in disc.discover_mcp_tools()
        # A bound progress sink makes the SDK add its own progressToken next to the plugin's keys.
        events = []
        with bound_tool_progress(
            lambda *a, **kw: events.append(kw), "mcp__meta__whoami", "call-1"
        ):
            raw = registry.dispatch(
                "mcp__meta__whoami", {"q": 1}, session_id="sess-1", task_id="task-1"
            )
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers

        shutdown_mcp_servers()
        _loop._stop_mcp_loop()

    seen = json.loads(json.loads(raw)["result"])
    assert seen["com.example/session"] == "sess-1"
    assert seen["com.example/call"] == "meta/whoami"
    assert seen["com.example/args-seen"] == ["q"]
    # The SDK's own progress token (the server names it progress_token), not the plugin's.
    assert seen.get("progress_token") not in (None, "forged")
    assert "modelcontextprotocol.io/forged" not in seen
    # The plugin's _meta does not displace the SDK's progress token: progress still arrives.
    assert [event["tool_call_id"] for event in events] == ["call-1"]


def test_no_mcp_request_middleware_leaves_the_call_unchanged():
    from tools import mcp_tool
    from tools import mcp_tool_handlers as handlers

    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text="ok", type="text")],
            isError=False,
            structuredContent=None,
            meta=None,
        )
    )

    def run_here(coro_or_factory, timeout=30):
        import asyncio

        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

        async def with_lock():
            server._rpc_lock = asyncio.Lock()
            return await coro

        return asyncio.run(with_lock())

    server = SimpleNamespace(session=session, _rpc_lock=None)
    with (
        patch.dict(mcp_tool._servers, {"srv": server}),
        patch.dict(mcp_tool._server_trust_levels, {"srv": "full"}),
        patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=run_here),
    ):
        assert json.loads(
            handlers._make_tool_handler("srv", "t", 30.0)({"a": 1}, session_id="s")
        ) == {"result": "ok"}

    session.call_tool.assert_awaited_once_with("t", arguments={"a": 1})
