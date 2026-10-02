"""Home Assistant identifier validation must finish before any HTTP request."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("aiohttp", reason="Home Assistant optional extra is not installed")

from agent.secret_scope import reset_secret_scope, set_secret_scope
from tools import homeassistant_tool
from tools.registry import registry


@contextmanager
def _api():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"entity_id": "light.demo", "state": "on"})
        def do_POST(self):
            self.reply([])
        def reply(self, payload):
            requests.append((self.command, self.path))
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = set_secret_scope({"HASS_URL": f"http://127.0.0.1:{server.server_port}", "HASS_TOKEN": "fixture-token"})
    try:
        yield requests
    finally:
        reset_secret_scope(token)
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("domain", sorted(homeassistant_tool._BLOCKED_DOMAINS))
def test_blocked_service_domain_cannot_be_hidden_behind_a_trailing_newline(domain):
    with _api() as requests:
        result = json.loads(registry.dispatch("ha_call_service", {"domain": domain + "\n", "service": "fixture"}))
        assert requests == []
        assert "error" in result


def test_valid_service_and_entity_work_but_identifiers_must_match_in_full():
    with _api() as requests:
        assert "error" not in json.loads(registry.dispatch("ha_call_service", {"domain": "light", "service": "turn_on"}))
        assert "error" not in json.loads(registry.dispatch("ha_get_state", {"entity_id": "light.demo"}))
        assert requests == [("POST", "/api/services/light/turn_on"), ("GET", "/api/states/light.demo")]
        requests.clear()
        results = []
        for tool, args in (
            ("ha_call_service", {"domain": "light", "service": "turn_on\n"}),
            ("ha_call_service", {"domain": "light", "service": "turn_on", "entity_id": "light.demo\n"}),
            ("ha_get_state", {"entity_id": "light.demo\n"}),
        ):
            results.append(json.loads(registry.dispatch(tool, args)))
        assert requests == []
        assert all("error" in result for result in results)
