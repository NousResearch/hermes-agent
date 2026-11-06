"""A reconnect that does not park republishes the new session's tool contract.

Real stdio MCP child, real discovery, real ``MCPServerTask`` run loop and registry; the
reconnect is the ``_reconnect_event`` path (manual refresh / OAuth recovery / transport drop
retried without parking), which keeps registrations across the outage.
"""

import concurrent.futures
import sys
import time

import pytest

_SERVER = '''import json, os, sys
contract = open(os.environ["CONTRACT_FILE"]).read().strip()
if contract == "A":
    tools = [{"name": "changing_tool", "description": "contract A",
              "inputSchema": {"type": "object", "properties": {"old": {"type": "string"}}}},
             {"name": "removed_tool", "description": "only in A", "inputSchema": {"type": "object"}}]
else:
    tools = [{"name": "changing_tool", "description": "contract B",
              "inputSchema": {"type": "object", "properties": {"new": {"type": "integer"}}}},
             {"name": "added_tool", "description": "only in B", "inputSchema": {"type": "object"}}]
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    result = {}
    if req["method"] == "initialize":
        result = {"protocolVersion": req["params"]["protocolVersion"],
                  "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                  "serverInfo": {"name": "contract", "version": "1"}}
    elif req["method"] == "tools/list":
        result = {"tools": tools}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
'''
P = "mcp__contract__"


def _server_config(tmp_path, contract_file):
    script = tmp_path / "contract_server.py"
    script.write_text(_SERVER, encoding="utf-8")
    contract_file.write_text("A", encoding="utf-8")
    return {"command": sys.executable, "args": [str(script)],
            "env": {"CONTRACT_FILE": str(contract_file)}, "timeout": 10}


def _view(scope):
    from tools.registry import registry
    return {name[len(P):]: (entry.schema.get("description"),
                            sorted((entry.schema.get("parameters") or {}).get("properties", {})))
            for name, entry in registry._slot(scope).items() if entry.toolset == "mcp-contract"}


def _wait(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _connection():
    from tools import mcp_tool
    from tools.mcp_tool_scope import _key_name
    with mcp_tool._lock:
        servers = [s for k, s in mcp_tool._servers.items() if _key_name(k) == "contract"]
    assert len(servers) == 1, servers
    return servers[0]


def _reconnect(server, contract_file, contract):
    from tools import mcp_tool
    contract_file.write_text(contract, encoding="utf-8")
    previous = server.session
    mcp_tool._mcp_loop.call_soon_threadsafe(server._reconnect_event.set)
    assert _wait(lambda: server.session is not None and server.session is not previous
                 and server._ready.is_set())


def _on_mcp_loop(fn):
    from tools import mcp_tool
    done = concurrent.futures.Future()
    mcp_tool._mcp_loop.call_soon_threadsafe(lambda: done.set_result(fn()))
    return done.result(5)


def test_quick_reconnect_publishes_changed_added_and_removed_tools(tmp_path):
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    contract_file = tmp_path / "contract"
    register_mcp_servers({"contract": _server_config(tmp_path, contract_file)})
    try:
        server = _connection()
        assert _view(None)["changing_tool"] == ("contract A", ["old"])
        assert "removed_tool" in _view(None)

        _reconnect(server, contract_file, "B")

        tools = _view(None)
        assert tools["changing_tool"] == ("contract B", ["new"])
        assert tools["added_tool"] == ("only in B", [])
        assert "removed_tool" not in tools
        assert sorted(server._registered_tool_names) == sorted(P + n for n in tools)
    finally:
        shutdown_mcp_servers()


def test_shared_connection_reconnect_refreshes_every_scope_with_its_own_policy(tmp_path):
    """Owner + adopter of one multiplexed connection, with different utility policy (not part
    of the route): both get the new contract, each keeps its own utilities, the teardown ledger
    stays their union, and an unchanged reconnect leaves the adopter untouched."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_constants import hermes_home_key
    from tools import mcp_tool
    from tools.mcp_tool_discovery import discover_mcp_tools
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import _key_name

    owner_home, worker_home = tmp_path / "default", tmp_path / "default" / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    contract_file = tmp_path / "contract"
    cfg = _server_config(tmp_path, contract_file)
    route = (f"    command: {cfg['command']}\n    args: ['{cfg['args'][0]}']\n"
             f"    env: {{CONTRACT_FILE: '{contract_file}'}}\n    timeout: 10\n")
    (owner_home / "config.yaml").write_text(
        "mcp_servers:\n  contract:\n" + route + "    tools: {prompts: false}\n", encoding="utf-8")
    (worker_home / "config.yaml").write_text(
        "mcp_servers:\n  contract:\n" + route + "    tools: {resources: false}\n", encoding="utf-8")
    owner, worker = hermes_home_key(owner_home), hermes_home_key(worker_home)
    owner_utils, worker_utils = {"list_resources", "read_resource"}, {"list_prompts", "get_prompt"}

    previous_multiplex = set_multiplex_active(True)
    try:
        for home in (owner_home, worker_home):
            with _profile_runtime_scope(home):
                discover_mcp_tools()
        server = _connection()
        assert set(_view(worker)) == {"changing_tool", "removed_tool"} | worker_utils  # adopted

        _reconnect(server, contract_file, "B")
        expected_tools = {"changing_tool": ("contract B", ["new"]), "added_tool": ("only in B", [])}
        assert _wait(lambda: "added_tool" in _view(worker))
        for scope, utils in ((owner, owner_utils), (worker, worker_utils)):
            tools = _view(scope)
            assert {n: tools[n] for n in expected_tools} == expected_tools
            assert set(tools) == set(expected_tools) | utils  # removed_tool gone, own policy only
        _connection()  # the adopter re-used the live connection, it did not open its own
        assert set(server._registered_tool_names) == {
            P + n for n in set(expected_tools) | owner_utils | worker_utils}

        before = (_view(owner), _view(worker))
        _reconnect(server, contract_file, "B")
        time.sleep(0.3)
        assert (_view(owner), _view(worker)) == before

        _on_mcp_loop(server._deregister_tools)  # park teardown is driven by the ledger alone
        assert _view(owner) == {} and _view(worker) == {}
        with mcp_tool._lock:
            assert [k for k in mcp_tool._servers if _key_name(k) == "contract"]
    finally:
        shutdown_mcp_servers()
        set_multiplex_active(previous_multiplex)
