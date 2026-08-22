"""The TUI gateway's MCP OAuth loopback relay must forward RFC 9207 ``iss``.

Desktop starts MCP OAuth through ``mcp.servers.oauth.start``, which binds the
loopback listener in ``tui_gateway/mcp_oauth_sessions.py``. mcp 2.0 validates
the authorization-response ``iss`` against the discovered metadata and rejects
a response that omits it whenever the server advertised
``authorization_response_iss_parameter_supported`` — every Cloudflare MCP
server does. A relay that parses only ``code``/``state``/``error`` therefore
breaks Desktop login with "Authorization response missing iss parameter",
even when the dashboard callback route already forwards it.

Real listener, real HTTP request — the parsing is the thing under test.
"""

import urllib.request

from tui_gateway import mcp_oauth_sessions


class _RecordingFlow:
    """Minimal stand-in for the flow object the relay feeds."""

    server_name = "test-server"

    def __init__(self):
        self.calls = []

    def deliver_callback(self, **kwargs):
        self.calls.append(kwargs)


def _get(port, query):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/callback?{query}", timeout=5
    ) as resp:
        return resp.status


def test_loopback_listener_forwards_iss():
    flow = _RecordingFlow()
    httpd = mcp_oauth_sessions._start_loopback_listener(flow)
    try:
        port = httpd.server_address[1]
        status = _get(port, "code=abc123&state=xyz&iss=https%3A%2F%2Fidp.example")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert len(flow.calls) == 1
    call = flow.calls[0]
    assert call["code"] == "abc123"
    assert call["state"] == "xyz"
    assert call["error"] is None
    # The parameter this test exists for.
    assert call["iss"] == "https://idp.example"


def test_loopback_listener_passes_iss_none_when_absent():
    """Providers that do not advertise the parameter must keep working:
    ``iss`` is forwarded as None rather than omitted from the call."""
    flow = _RecordingFlow()
    httpd = mcp_oauth_sessions._start_loopback_listener(flow)
    try:
        port = httpd.server_address[1]
        assert _get(port, "code=abc123&state=xyz") == 200
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert flow.calls == [
        {"code": "abc123", "state": "xyz", "error": None, "iss": None}
    ]
