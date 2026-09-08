"""Exercise the real connector resolution/dispatch path with an isolated identity."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


def test_anonymous_connector_round_trip_needs_no_subscription_preflight(tmp_path, monkeypatch):
    from hermes_cli.anon_auth import ANON_AUTH_METHOD
    from tools import tool_backend_helpers
    from tools.tool_gateway.config import connectors_available

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    (tmp_path / "config.yaml").write_text(
        "nous:\n  guest: true\ntools:\n  connectors:\n    enabled: true\n"
        "  tool_search:\n    defer: []\n"
    )
    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {
        "auth_method": ANON_AUTH_METHOD, "access_token": "anonymous-test-token",
        "anon_token": "isolated-test-identity", "expires_at": "2099-01-01T00:00:00Z",
    }}}))
    preflights = []
    monkeypatch.setattr(tool_backend_helpers, "managed_nous_tools_enabled",
                        lambda: preflights.append(True) or False)
    assert connectors_available()
    assert not preflights

    requests = []
    schema = {"connector": "gmail", "tool": "GMAIL_LIST_MESSAGES", "description": "List messages",
              "input_schema": {"type": "object", "properties": {}}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization"), None))
            self.reply({"items": [{"connector": "gmail", "enabled": True, "connected": True}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            if self.path.endswith("/search"):
                self.reply({"results": [{"index": 1, "use_case": "list mail",
                                         "tools": ["GMAIL_LIST_MESSAGES"]}],
                            "schemas": {"GMAIL_LIST_MESSAGES": schema}, "connections": []})
            elif self.path.endswith("/schemas"):
                self.reply({"schemas": {"GMAIL_LIST_MESSAGES": schema}, "not_found": []})
            else:
                # Policy still belongs to the server, even for an anonymous identity.
                self.reply({"results": [{"index": 0, "connector": "gmail", "tool": "GMAIL_LIST_MESSAGES",
                                         "error": {"code": "TOOL_NOT_ALLOWED", "message": "server policy"}}],
                            "successCount": 0, "errorCount": 1, "totalCount": 1})

        def reply(self, body):
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("CONNECTOR_GATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        import model_tools
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        defs = model_tools.get_tool_definitions(enabled_toolsets=["connections"], quiet_mode=True,
                                                skip_tool_search_assembly=True)
        assert "manage_connections" in {t["function"]["name"] for t in defs}
        assembly = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw({"defer": []}))
        assert assembly.activated
        assert {t["function"]["name"] for t in assembly.tool_defs} == {
            "manage_connections", "tool_search", "tool_describe", "tool_call"}
        name = "connectors__gmail__LIST_MESSAGES"
        call = lambda tool, args: json.loads(model_tools.handle_function_call(
            tool, args, enabled_toolsets=["connections"], session_id="isolated"))
        assert name in call("tool_search", {"queries": ["list mail"]})["tools"]
        assert name in call("tool_describe", {"names": [name]})["tools"]
        assert call("manage_connections", {"action": "status"})["connectors"][0]["connected"]
        result = call("tool_call", {"calls": [{"name": name, "arguments": {}}]})
        assert result["results"][0]["error"]["code"] == "TOOL_NOT_ALLOWED"
        assert len(requests) == 4
        assert all(auth == "Bearer anonymous-test-token" for _, auth, _ in requests)
        assert not preflights
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("guest_enabled,connectors_enabled", [(False, True), (True, False)])
def test_connector_off_switches_never_send_a_guest_bearer(tmp_path, monkeypatch, guest_enabled, connectors_enabled):
    from hermes_cli.anon_auth import ANON_AUTH_METHOD
    from tools.tool_gateway.config import connectors_available
    from tools import tool_backend_helpers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"nous:\n  guest: {str(guest_enabled).lower()}\ntools:\n  connectors:\n"
        f"    enabled: {str(connectors_enabled).lower()}\n"
    )
    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {
        "auth_method": ANON_AUTH_METHOD, "access_token": "anonymous-test-token",
    }}}))
    monkeypatch.setattr(tool_backend_helpers, "managed_nous_tools_enabled", lambda: False)
    assert not connectors_available()
