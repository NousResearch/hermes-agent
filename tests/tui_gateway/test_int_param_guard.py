"""Malformed numeric RPC parameters return a response without escaping the handler."""

import pytest
import tui_gateway.server as server


@pytest.fixture(autouse=True)
def isolated_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *args, **kwargs: None)


@pytest.mark.parametrize("method,key,bad,valid", [
    (method, key, value, valid)
    for method, key in [("session.create", "cols"), ("session.list", "limit"),
                        ("spawn_tree.list", "limit"), ("llm.oneshot", "max_tokens"),
                        ("pet.cells", "cols"), ("pet.generate", "count")]
    for value, valid in [({}, False), ([1], False), (None, True), ("bad", False),
                         (float("inf"), False), (1.5, False), ("120", True), (100, True)]
    if method in {"session.create", "session.list", "spawn_tree.list"} or not valid
])
def test_numeric_defaults_through_rpc(method, key, bad, valid):
    response = server.handle_request({
        "jsonrpc": "2.0", "id": "r1", "method": method, "params": {key: bad},
    })
    if not valid:
        assert response["error"]["code"] == 4000
        assert not server._sessions
        return
    assert "result" in response, response
    if method == "session.create":
        assert server._sessions[response["result"]["session_id"]]["cols"] == (80 if bad is None else int(bad))


@pytest.mark.parametrize("value,expected", [({}, None), ([1], None), (None, None), ("bad", None),
                                            (float("inf"), None), ("120", 120), (100, 100)])
def test_terminal_resize_validates_before_mutating_session(value, expected):
    server._sessions["resize"] = {"session_key": "resize", "history": [], "cols": 80}
    response = server.handle_request({
        "jsonrpc": "2.0", "id": "r2", "method": "terminal.resize",
        "params": {"session_id": "resize", "cols": value},
    })
    if expected is None:
        assert response["error"]["code"] == 4000
        assert server._sessions["resize"]["cols"] == 80
    else:
        assert response["result"]["cols"] == expected
        assert server._sessions["resize"]["cols"] == expected
