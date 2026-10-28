"""Real stdio MCP server: the catalog the surfaces answer with follows the connection.

Both claims here need a real transport and the production sender: a reconnect must replace
the catalog, and a listing the user asked for must read what the server serves *now* rather
than what was read when the connection was opened.
"""

import json
import sys

import pytest

from tests.waiting import wait_until

_SERVER = '''
import json, sys
catalog_path = sys.argv[1]
for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "serverInfo": {"name": "catalog", "version": "1"},
                  "capabilities": {"resources": {"listChanged": True},
                                   "extensions": {"io.modelcontextprotocol/skills": {}}}}
    elif method == "skills/list":
        with open(catalog_path, encoding="utf-8") as source:
            skills = json.load(source)
        result = {"skills": skills, "ttlMs": 600000, "cacheScope": "private"}
    elif method == "ping":
        result = {}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                          "error": {"code": -32601, "message": "not implemented"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
'''


@pytest.fixture
def live_catalog(tmp_path, monkeypatch):
    """A connected real stdio skills server whose published catalog the test can rewrite."""
    from tests.fakes.fake_mcp_skills import FakeSkill
    from tools import mcp_skills_manager as catalogs
    from tools.mcp_tool_discovery import _discover_and_register_server
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_loop import _ensure_mcp_loop, _run_on_mcp_loop

    monkeypatch.setattr(catalogs, "_MANAGER", catalogs.SkillsOverMcpManager())
    monkeypatch.setattr(catalogs, "current_scope_id", lambda: "")
    script, catalog_file = tmp_path / "catalog_server.py", tmp_path / "catalog.json"
    script.write_text(_SERVER, encoding="utf-8")
    publish = lambda *skills: catalog_file.write_text(
        json.dumps([skill.entry() for skill in skills]), encoding="utf-8")
    kept = FakeSkill(uri="skill://team/kept/SKILL.md", name="kept")
    publish(FakeSkill(uri="skill://team/removed/SKILL.md", name="removed"), kept)
    config = {"command": sys.executable, "args": ["-I", str(script), str(catalog_file)],
              "skills": {"enabled": True}, "connect_timeout": 15}
    _ensure_mcp_loop()
    _run_on_mcp_loop(lambda: _discover_and_register_server("catalog-live", config), timeout=30)
    try:
        yield {"manager": catalogs.get_manager(), "scope": "", "server": "catalog-live",
               "publish": publish, "kept": kept, "run": _run_on_mcp_loop}
    finally:
        shutdown_mcp_servers(names={"catalog-live"})


def test_live_stdio_reconnect_replaces_the_catalog(live_catalog):
    from tools.mcp_tool_common import _core

    manager, scope, kept = live_catalog["manager"], live_catalog["scope"], live_catalog["kept"]
    assert {entry.name for entry in manager.entries(scope)} == {"removed", "kept"}
    first = manager.snapshot(live_catalog["server"], scope)
    live_catalog["publish"](kept)
    with _core._lock:
        server = next(live for live in _core._servers.values() if live.name == live_catalog["server"])
    live_catalog["run"](lambda: _reconnect(server), timeout=5)
    wait_until(lambda: {entry.name for entry in manager.entries(scope)} == {"kept"},
               timeout=30, what="reconnected catalog")
    assert manager.snapshot(live_catalog["server"], scope).connection_generation > first.connection_generation


async def _reconnect(server):
    server._reconnect_event.set()


def test_a_listing_reads_the_catalog_the_server_serves_now(live_catalog, monkeypatch):
    """The reported bug: the listing answered from the catalog read when the connection opened."""
    from tools import skills_tool_mcp

    manager, scope, kept = live_catalog["manager"], live_catalog["scope"], live_catalog["kept"]
    monkeypatch.setattr(skills_tool_mcp, "known_server_labels", lambda: {live_catalog["server"]})
    monkeypatch.setattr(skills_tool_mcp, "_scope_id", lambda: scope)
    assert {row["name"] for row in skills_tool_mcp.list_rows(scope)} == {"catalog-live::team/removed",
                                                                        "catalog-live::team/kept"}
    live_catalog["publish"](kept)

    verified, problems = skills_tool_mcp.refresh_catalogs()

    assert verified == [live_catalog["server"]] and problems == []
    assert {entry.name for entry in manager.entries(scope)} == {"kept"}


def test_a_listing_over_a_connection_the_server_dropped_verifies_nothing(live_catalog, monkeypatch):
    """A connect-time snapshot must not outlive the connection it was read over."""
    from tools import skills_tool_mcp
    from tools.mcp_tool_common import _core

    manager, scope = live_catalog["manager"], live_catalog["scope"]
    monkeypatch.setattr(skills_tool_mcp, "known_server_labels", lambda: {live_catalog["server"]})
    monkeypatch.setattr(skills_tool_mcp, "_scope_id", lambda: scope)
    monkeypatch.setattr(skills_tool_mcp, "_connect_servers", lambda servers: None)
    with _core._lock:
        server = next(live for live in _core._servers.values() if live.name == live_catalog["server"])
    server.session = None  # what every teardown leaves behind, before anything re-lists

    verified, problems = skills_tool_mcp.refresh_catalogs(timeout=0.5)

    assert verified == [] and problems and problems[0]["server"] == live_catalog["server"]
    assert manager.entries(scope) != [], "the snapshot is retired by the lifecycle, not by a listing"
