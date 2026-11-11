"""Bind catalog lifecycle events to the scopes served by one physical MCP transport."""

import asyncio

from tools.mcp_skills_manager import (
    get_manager,
    mcp_server_skills_enabled,
    publish_catalog_after_connect,
)


def adopt_transport_catalog(server, config, scope):
    """A shared transport is not consent to share its owner's catalog."""
    from tools.mcp_skills_manager import _mcp_loop
    manager = get_manager()
    if not mcp_server_skills_enabled(config):
        manager.mark_disconnected(server.name, scope, "skills disabled")
        inst = manager._instance(server.name, scope)
        inst.transport = None
        return
    loop = _mcp_loop()
    if loop is not None:
        asyncio.run_coroutine_threadsafe(
            publish_catalog_after_connect(server.name, config, server, scope_id=scope), loop)


def _catalog_scopes(server):
    manager = get_manager()
    scopes = {view.scope_id for view in manager.instances()
              if manager._instance(view.server_name, view.scope_id).transport is server}
    if mcp_server_skills_enabled(server._config):
        from tools.mcp_tool_common import _core
        with _core._lock:
            owners = [_core._server_registry_scope(key) or ""
                      for key, live in _core._servers.items() if live is server]
        scopes.update(owners)
    return scopes


async def publish_transport_catalogs(server):
    await asyncio.gather(*(publish_catalog_after_connect(
        server.name, {"skills": {"enabled": True}}, server, scope_id=scope)
        for scope in _catalog_scopes(server)))


def retire_transport_catalogs(server):
    manager = get_manager()
    for scope in _catalog_scopes(server):
        manager.mark_disconnected(server.name, scope, "transport closed")


def refresh_transport_catalogs(server):
    from tools import mcp_skills_manager
    manager = get_manager()
    for scope in _catalog_scopes(server):
        manager.request_refresh(server.name, scope, reason="resources/list_changed",
                                min_interval=mcp_skills_manager.MIN_REFRESH_INTERVAL_SECONDS, defer=True)
