import json
import sys

import pytest

from tools.computer_use import cua_backend_driver
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession


@pytest.fixture
def mcp_session(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    server = tmp_path / "server.py"
    server.write_text('''
import json
import os
import sys
from pathlib import Path

journal = Path(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method, params = request["method"], request.get("params", {})
    response = {"jsonrpc": "2.0", "id": request["id"]}
    if method == "initialize":
        result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "crash-fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": []}
    elif method == "tools/call":
        name = params["name"]
        prior = journal.read_text(encoding="utf-8") if journal.exists() else ""
        with journal.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"name": name, "args": params.get("arguments", {}), "pid": os.getpid()}) + "\\n")
        if params.get("arguments", {}).get("reject"):
            response["error"] = {"code": -32602, "message": "Connection closed"}
            print(json.dumps(response), flush=True)
            continue
        if name != "start_session" and not any(json.loads(row)["name"] != "start_session" for row in prior.splitlines()):
            os._exit(0)
        result = {"content": [{"type": "text", "text": json.dumps({"ok": True})}], "isError": False}
    else:
        result = {}
    response["result"] = result
    print(json.dumps(response), flush=True)
''', encoding="utf-8")
    journal = tmp_path / "requests.jsonl"
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: sys.executable)
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation",
                        lambda command: (command, [str(server), str(journal)]))
    bridge = _AsyncBridge()
    session = _CuaDriverSession(bridge)
    try:
        session.start()
        yield session, journal
    finally:
        try:
            session.stop()
        finally:
            bridge.stop()


@pytest.mark.parametrize("operation", ["list_apps", "click"])
def test_mcp_crash_recovers_reads_without_replaying_mutations(mcp_session, operation):
    session, journal = mcp_session
    resets = []
    session.set_transport_reset_callback(lambda: resets.append(True))
    session.call_tool("start_session", {"session": "isolated-crash-test"})

    result = session.call_tool(operation, {})

    if operation == "list_apps":
        assert result["isError"] is False
    else:
        assert result["isError"] is True
        assert result["structuredContent"]["code"] == "transport_outcome_unknown"
        assert result["structuredContent"]["next_step"] == "fresh_state"
    calls = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [call["name"] for call in calls] == (
        ["start_session", operation, "start_session", operation]
        if operation == "list_apps" else ["start_session", operation, "start_session"])
    assert calls[0]["pid"] == calls[1]["pid"]
    assert calls[0]["pid"] != calls[2]["pid"]
    assert calls[0]["args"] == calls[2]["args"]
    assert resets == [True]
    assert session.call_tool("list_apps", {})["isError"] is False
    later_calls = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert later_calls[:-1] == calls
    assert later_calls[-1]["name"] == "list_apps"
    assert later_calls[-1]["pid"] == calls[2]["pid"]
    assert resets == [True]


def test_mcp_application_error_is_not_a_closed_transport(mcp_session):
    from mcp.shared.exceptions import MCPError
    from mcp_types.jsonrpc import INVALID_PARAMS

    session, journal = mcp_session
    resets = []
    session.set_transport_reset_callback(lambda: resets.append(True))
    with pytest.raises(MCPError) as raised:
        session.call_tool("list_apps", {"reject": True})
    assert raised.value.code == INVALID_PARAMS
    assert str(raised.value) == "Connection closed"
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1
    assert resets == []
