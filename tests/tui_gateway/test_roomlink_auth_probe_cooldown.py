"""Rejected RoomLink credentials must not generate one HTTP request per tick."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tui_gateway import hosted_room_peer_http as peer_http
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.fixture
def endpoint():
    calls = []
    response_status = {"value": 401}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.command, self.path, self.headers.get("Authorization")))
            status = response_status["value"]
            payload = (
                {"error": {"code": "invalid_room_grant"}}
                if status != 200
                else {"catalog": {"ready": True}}
            )
            body = (
                b"x" * response_status["body_bytes"]
                if "body_bytes" in response_status
                else json.dumps(payload).encode()
            )
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.do_GET()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, response_status
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/v1/room-members/capabilities"), ("POST", "/v1/room-members/grants/refresh")],
)
def test_rejected_probe_waits_but_replacement_grant_is_immediate(endpoint, method, path):
    url, calls, response_status = endpoint
    now = [10.0]
    client = PeerRunsHTTPClient(base_url=url, api_key="", clock=lambda: now[0])

    def request(grant):
        return client._request(path, method=method, room_grant=grant)

    for _ in range(8):
        with pytest.raises(PeerRunsHTTPError) as failure:
            request("rejected.room.grant")
        assert failure.value.needs_reauthorization
        assert not failure.value.ambiguous
        now[0] += 0.25
    assert len(calls) == 1

    response_status["value"] = 200
    assert request("replacement.room.grant")["catalog"]["ready"]
    assert len(calls) == 2
    now[0] += 61
    assert request("rejected.room.grant")["catalog"]["ready"]
    assert len(calls) == 3


def test_auth_cooldown_never_blocks_stop_status_or_transient_recovery(endpoint):
    url, calls, response_status = endpoint
    client = PeerRunsHTTPClient(base_url=url, api_key="", clock=lambda: 10.0)
    grant = "rejected.room.grant"
    with pytest.raises(PeerRunsHTTPError):
        client._request("/v1/room-members/capabilities", room_grant=grant)
    response_status["value"] = 200
    client._request("/v1/runs/existing/stop", method="POST", room_grant=grant)
    client._request("/v1/runs/existing", room_grant=grant)
    assert len(calls) == 3

    response_status["value"] = 503
    for _ in range(2):
        with pytest.raises(PeerRunsHTTPError) as failure:
            client._request("/v1/room-members/capabilities", room_grant="temporary.room.grant")
        assert failure.value.retryable
    assert len(calls) == 5


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("failure_kind", ["oversized", "deadline"])
def test_rejected_error_body_is_bounded_and_still_cools_down(endpoint, monkeypatch, status, failure_kind):
    url, calls, response_status = endpoint
    response_status["value"] = status
    if failure_kind == "oversized":
        response_status["body_bytes"] = peer_http.MAX_PEER_ERROR_RESPONSE_BYTES + 1
    else:
        def deadline(*_args, **_kwargs):
            raise peer_http._PeerResponseDeadlineExceeded

        monkeypatch.setattr(peer_http, "_read_bounded_response", deadline)
    client = PeerRunsHTTPClient(base_url=url, api_key="", clock=lambda: 10.0)
    for _ in range(3):
        with pytest.raises(PeerRunsHTTPError) as failure:
            client._request("/v1/room-members/grants/refresh", method="POST", room_grant="rejected.room.grant")
        assert failure.value.status_code == status
        assert failure.value.retryable is (failure_kind == "deadline")
        assert not failure.value.ambiguous
        assert failure.value.error_code is None
    assert len(calls) == 1
