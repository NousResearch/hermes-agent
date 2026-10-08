"""An off-scope tools/list_changed refresh cannot mutate an editor's frozen tools."""
from types import SimpleNamespace

import pytest


def test_editor_connection_scope_fences_later_schema_refresh(monkeypatch):
    from tools import mcp_tool as core
    from tools.mcp_tool_registration import _register_server_tools
    from tools.registry import registry, session_tool_scope
    from hermes_state_runtime import RuntimeStoreError

    server = core.MCPServerTask('editor-freeze')
    server._tools = [SimpleNamespace(name='echo', description='initial',
                                    inputSchema={'type': 'object', 'properties': {}})]
    key = ('editor-session:review-freeze', server.name)
    monkeypatch.setitem(core._servers, key, server)
    monkeypatch.setitem(core._server_scope_keys, key, key[0])
    config = {'tools': {'resources': False, 'prompts': False}}
    with session_tool_scope(key[0]):
        names = _register_server_tools(server.name, server, config)
    assert names
    try:
        server._tools[0].description = 'changed after session creation'
        with pytest.raises(RuntimeStoreError, match='acp_mcp_schema_changed'):
            _register_server_tools(server.name, server, config)
    finally:
        for name in names:
            registry.deregister(name, scope=key[0])
