"""Routine discovery must not wake a stdio server that is dormant on purpose (idle/lifetime recycle)."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def known_server(monkeypatch):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as discovery, mcp_tool_loop as loop

    signalled = []
    monkeypatch.setattr(loop, "_signal_reconnect", lambda srv: signalled.append(srv) or True)
    name, config = "dormant_wake_fixture", {"command": "true"}
    with core._lock:
        key = discovery._resolve_server_key(name, core._mcp_registry_scope(), current=False)

    def install(recycled_reason):
        server = SimpleNamespace(session=None, _recycled_reason=recycled_reason, _auth_park=None)
        with core._lock:
            core._servers[key] = server
        return server

    yield SimpleNamespace(install=install, select=lambda: discovery._select_new_servers({name: config}),
                          signalled=signalled)
    with core._lock:
        core._servers.pop(key, None)


def test_dormant_recycled_server_is_left_asleep(known_server):
    known_server.install("idle_timeout_seconds")
    assert known_server.select() == {}
    assert known_server.signalled == []


def test_server_without_session_for_another_reason_is_still_nudged(known_server):
    server = known_server.install(None)
    assert known_server.select() == {}
    assert known_server.signalled == [server]
