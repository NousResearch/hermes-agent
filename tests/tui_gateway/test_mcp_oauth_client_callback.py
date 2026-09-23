"""Tests for the client-side callback (remote-backend) variant of the
session-backed MCP OAuth flow (tui_gateway/mcp_oauth_sessions.py).

Covers the three seams added for remote Desktop backends:
  - _validate_client_redirect_uri: loopback-only allowlist for the
    client-supplied redirect URI (rejects public hosts/schemes so a gateway
    never pins an attacker-controlled redirect into a DCR registration);
  - start_flow(client_redirect_uri=...): no gateway-side listener is bound and
    the flow's redirect_uri is pinned to the client's listener;
  - deliver_callback_flow: relays a client-captured code/state into the flow
    with the SAME state verification as the loopback path (wrong state
    rejected, replay rejected, unknown session rejected).
"""


import pytest

from hermes_constants import get_hermes_home
from tools.connectors import mcp_oauth
from tools.connectors.mcp_oauth import _validate_client_redirect_uri
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions
from tui_gateway.mcp_oauth_sessions import deliver_callback_flow


# ---------------------------------------------------------------------------
# _validate_client_redirect_uri
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("http://127.0.0.1:8412/callback", "http://127.0.0.1:8412/callback"),
        ("http://localhost:60000/callback", "http://localhost:60000/callback"),
        # Path defaulting
        ("http://127.0.0.1:9999", "http://127.0.0.1:9999/callback"),
        # Surrounding whitespace tolerated
        ("  http://127.0.0.1:8412/callback  ", "http://127.0.0.1:8412/callback"),
    ],
)
def test_validate_accepts_loopback_http(uri, expected):
    assert _validate_client_redirect_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "https://127.0.0.1:8412/callback",  # https is not a native loopback
        "http://evil.example.com:8412/callback",  # public host
        "http://192.168.1.10:8412/callback",  # LAN host
        "http://127.0.0.1/callback",  # no port
        "http://user:pass@127.0.0.1:8412/callback",  # credentials
        "javascript:alert(1)",
        "",
        "not a url",
    ],
)
def test_validate_rejects_non_loopback(uri):
    with pytest.raises(ValueError):
        _validate_client_redirect_uri(uri)


# ---------------------------------------------------------------------------
# start_flow with client_redirect_uri: no gateway listener, URI pinned
# ---------------------------------------------------------------------------


def _fake_worker_publishes_url(monkeypatch, state="teststate123"):
    """Replace the OAuth worker with a stub that publishes an authorize URL
    carrying *state* and then waits for the callback like the real worker's
    SDK does."""

    def worker(hermes_home, server_name, cfg, reconnect_live, *, flow, on_done=None, **_card_options):
        import asyncio

        asyncio.run(
            flow.publish_authorization_url(
                f"https://as.example.com/authorize?client_id=x&state={state}"
            )
        )
        # Wait for the callback (delivered by the test), then approve.
        try:
            asyncio.run(flow.wait_for_callback(timeout=5))
            flow.mark_approved()
        except Exception as exc:  # pragma: no cover - failure surface
            flow.mark_error(str(exc))
        finally:
            flow.mark_worker_done()
            if on_done is not None:
                on_done()

    monkeypatch.setattr(mcp_oauth_sessions, "run_worker", worker)
    return worker


def test_start_flow_client_redirect_skips_gateway_listener(monkeypatch):
    worker = _fake_worker_publishes_url(monkeypatch)

    bound = []
    real_receiver = mcp_oauth._start_loopback_receiver
    monkeypatch.setattr(
        mcp_oauth,
        "_start_loopback_receiver",
        lambda flow: bound.append(flow) or real_receiver(flow),
    )

    result = mcp_oauth_sessions.start_flow(
        str(get_hermes_home()),
        "clicky",
        {"url": "https://mcp.example.com/mcp", "auth": "oauth"},
        client_redirect_uri="http://127.0.0.1:8412/callback",
    )

    assert result["session_id"]
    assert result["auth_url"].startswith("https://as.example.com/authorize")
    assert bound == [], "gateway listener must NOT be bound with a client redirect"

    rec = mcp_oauth_sessions._sessions[result["session_id"]]
    assert rec["httpd"] is None
    assert rec["flow"].redirect_uri == "http://127.0.0.1:8412/callback"

    # Cleanup: deliver the callback so the stub worker thread exits.
    deliver_callback_flow(
        result["session_id"], "clicky", code="authcode", state="teststate123"
    )
    rec["flow"]._worker_done.wait(5)

    # The connection-card path does not require a dashboard web server: it binds the same backend
    # receiver, registers the flow for callback relay, and carries the SSH paste hint in detail.
    import hermes_cli.mcp_config as mcp_config

    monkeypatch.setattr(
        mcp_config,
        "_get_mcp_servers",
        lambda: {"cardy": {"url": "https://mcp.example.com/mcp", "auth": "oauth"}},
    )
    monkeypatch.setattr(mcp_oauth, "run_worker", worker)
    monkeypatch.setenv("SSH_CLIENT", "192.0.2.1 12345 22")
    attempt = mcp_oauth.start("cardy")
    assert attempt.flow.redirect_uri.startswith("http://127.0.0.1:")
    assert attempt.flow.flow_id in mcp_oauth_sessions._sessions
    deliver_callback_flow(
        attempt.flow.flow_id, "cardy", code="authcode", state="teststate123"
    )
    attempt.flow._worker_done.wait(5)

    # A pre-registered client owns its pinned listener inside the SDK; the receiver picker must
    # publish that URI without attempting a second bind.
    pinned = DashboardOAuthFlow(
        "pinned", "asana", None, str(get_hermes_home()), ""
    )
    pinned_bound = []
    monkeypatch.setattr(
        mcp_oauth, "_start_loopback_receiver", lambda flow: pinned_bound.append(flow)
    )
    assert mcp_oauth.choose_callback_receiver(
        pinned,
        {"oauth": {"client_id": "asana-client", "redirect_host": "localhost",
                   "redirect_port": 27890}},
    ) is None
    assert pinned.redirect_uri == "http://localhost:27890/callback"
    assert pinned_bound == []


def test_start_flow_rejects_bad_client_redirect(monkeypatch):
    _fake_worker_publishes_url(monkeypatch)
    with pytest.raises(ValueError):
        mcp_oauth_sessions.start_flow(
            str(get_hermes_home()),
            "clicky2",
            {"url": "https://mcp.example.com/mcp", "auth": "oauth"},
            client_redirect_uri="https://evil.example.com/callback",
        )
    # No session must be left behind by a rejected start.
    assert all(
        r["server_name"] != "clicky2" for r in mcp_oauth_sessions._sessions.values()
    )


# ---------------------------------------------------------------------------
# deliver_callback_flow: relay accept/reject semantics
# ---------------------------------------------------------------------------


def _make_session(session_id="sess-relay-1", server="hosp", state="s3cr3tstate"):
    flow = DashboardOAuthFlow(
        flow_id=session_id,
        server_name=server,
        profile=None,
        hermes_home=str(get_hermes_home()),
        redirect_uri="http://127.0.0.1:9000/callback",
    )
    # Pin the expected state the way publish_authorization_url does.
    import asyncio

    asyncio.run(
        flow.publish_authorization_url(
            f"https://as.example.com/authorize?state={state}"
        )
    )
    rec = {
        "session_id": session_id,
        "server_name": server,
        "hermes_home": str(get_hermes_home()),
        "flow": flow,
        "httpd": None,
        "created_at": __import__("time").time(),
    }
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions[session_id] = rec
    return flow


def teardown_function(_fn):
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions.clear()


def test_deliver_callback_accepts_matching_state():
    flow = _make_session()
    out = deliver_callback_flow("sess-relay-1", "hosp", code="abc", state="s3cr3tstate")
    assert out == {"ok": True, "session_id": "sess-relay-1"}
    assert flow._callback == ("abc", "s3cr3tstate", None)


def test_oauth_callback_rpc_relays_iss():
    """The gateway ``mcp.servers.oauth.callback`` RPC accepts ``iss`` under the extra=forbid contract
    and forwards it to the flow; the desktop renderer always sends the key (possibly null)."""
    import tui_gateway.server as srv
    from tui_gateway.contracts import registry as contracts

    flow = _make_session()
    contract = contracts.METHODS["mcp.servers.oauth.callback"]
    params = {"session_id": "sess-relay-1", "name": "hosp", "code": "abc", "state": "s3cr3tstate",
              "iss": "https://as.example.com"}
    params, problem = contracts.validate_params(contract, params)
    assert problem is None
    out = srv._methods["mcp.servers.oauth.callback"](1, params)
    assert out["result"]["ok"] is True
    assert flow._callback == ("abc", "s3cr3tstate", "https://as.example.com")


def test_loopback_listener_forwards_iss():
    """The gateway-hosted loopback listener parses ``iss`` off the redirect rather than dropping it."""
    import urllib.request

    flow = _make_session(session_id="sess-relay-loop", server="loopy", state="loopstate")
    httpd = mcp_oauth._start_loopback_receiver(flow)
    try:
        port = httpd.server_address[1]
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=abc&state=loopstate&iss=https%3A%2F%2Fas.example.com",
            timeout=5,
        ).read()
    finally:
        httpd.shutdown()
    assert flow._callback == ("abc", "loopstate", "https://as.example.com")


def test_deliver_callback_rejects_state_mismatch():
    _make_session()
    out = deliver_callback_flow("sess-relay-1", "hosp", code="abc", state="WRONG")
    assert out["ok"] is False
    assert "state" in out["error_message"].lower()


def test_deliver_callback_rejects_replay():
    _make_session()
    first = deliver_callback_flow(
        "sess-relay-1", "hosp", code="abc", state="s3cr3tstate"
    )
    assert first["ok"] is True
    second = deliver_callback_flow(
        "sess-relay-1", "hosp", code="abc", state="s3cr3tstate"
    )
    assert second["ok"] is False


def test_deliver_callback_unknown_session_and_name_mismatch():
    _make_session()
    assert deliver_callback_flow("nope", "hosp", code="a", state="s")["ok"] is False
    out = deliver_callback_flow("sess-relay-1", "other-server", code="a", state="s")
    assert out["ok"] is False
    assert "mismatch" in out["error_message"]


def test_deliver_callback_propagates_provider_error():
    flow = _make_session()
    out = deliver_callback_flow(
        "sess-relay-1", "hosp", code=None, state="s3cr3tstate", error="access_denied"
    )
    assert out["ok"] is True  # accepted; the flow records the provider error

    async def _check():
        with pytest.raises(RuntimeError, match="access_denied"):
            await flow.wait_for_callback(timeout=1)

    import asyncio

    asyncio.run(_check())
