"""Hosted-dashboard bridge for MCP OAuth browser callbacks."""

import asyncio
import threading

import pytest


def test_dashboard_flow_exposes_authorization_url_and_accepts_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-1",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-1",
    )

    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))
    assert flow.snapshot() == {
        "flow_id": "flow-1",
        "server_name": "reports",
        "status": "authorization_required",
        "authorization_url": "https://idp.example/authorize?state=s1",
        "error": None,
    }

    flow.deliver_callback(code="code-1", state="s1", error=None)
    assert asyncio.run(flow.wait_for_callback()) == ("code-1", "s1", None)


def test_dashboard_flow_preserves_rfc9207_iss():
    """RFC 9207 ``iss`` survives the callback bridge: mcp 2.x rejects an authorization response
    that omits it when the authorization server advertised support (Cloudflare, Resend)."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-iss",
        server_name="cloudflare",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-iss",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    flow.deliver_callback(code="code-1", state="s1", error=None, iss="https://mcp.cloudflare.com")
    assert asyncio.run(flow.wait_for_callback()) == ("code-1", "s1", "https://mcp.cloudflare.com")


def test_dashboard_flow_accepts_only_one_concurrent_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-race",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-race",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=state"))

    start = threading.Barrier(3)
    outcomes: list[str] = []

    def deliver(code: str) -> None:
        start.wait()
        try:
            flow.deliver_callback(code=code, state="state", error=None)
            outcomes.append("accepted")
        except ValueError:
            outcomes.append("rejected")

    workers = [threading.Thread(target=deliver, args=(code,)) for code in ("one", "two")]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join()

    assert sorted(outcomes) == ["accepted", "rejected"]


def test_mcp_oauth_helpers_use_dashboard_flow_without_loopback_port():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _make_callback_waiter,
        _make_redirect_handler,
    )

    flow = DashboardOAuthFlow(
        flow_id="flow-4",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-4",
    )
    cfg = {}
    with dashboard_oauth_flow(flow):
        assert _configure_callback_port(cfg, HermesTokenStorage("reports")) == 0
        metadata = _build_client_metadata(cfg)
        assert str(metadata.redirect_uris[0]) == flow.redirect_uri

        asyncio.run(
            _make_redirect_handler(0)(
                "https://idp.example/authorize?state=state-4"
            )
        )
        flow.deliver_callback(code="code-4", state="state-4", error=None)
        # mcp 2.0's callback_handler contract returns an
        # AuthorizationCodeResult, not the legacy (code, state) tuple.
        result = asyncio.run(_make_callback_waiter(0)())
        assert (result.code, result.state) == ("code-4", "state-4")

    assert flow.authorization_url == "https://idp.example/authorize?state=state-4"


def test_failed_reauth_rollback_preserves_newer_oauth_state(tmp_path, monkeypatch):
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("reports")
    storage._tokens_path().parent.mkdir(parents=True)
    storage._tokens_path().write_text("OLD", encoding="utf-8")
    backup = storage.snapshot()
    storage.remove()

    storage._tokens_path().write_text("FRESH", encoding="utf-8")
    storage.restore(backup, only_if_absent=True)

    assert storage._tokens_path().read_text(encoding="utf-8") == "FRESH"


def test_preregistered_pinned_redirect_port_keeps_loopback_listener_under_dashboard_flow(tmp_path, monkeypatch):
    """A no-DCR entry (``client_id`` + ``redirect_port``, e.g. the shipped Asana manifest) registered
    ``http://localhost:<port>/callback`` with the vendor, which matches redirect URLs exactly. The
    dashboard/Desktop Authorize button must therefore keep that loopback URI and listener — the
    dashboard flow only publishes the authorization URL — instead of forcing its own callback URL."""
    import socket
    import urllib.request

    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import (
        HermesTokenStorage,
        _build_client_metadata,
        _configure_callback_port,
        _make_callback_waiter,
        _make_redirect_handler,
        force_interactive_oauth,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    flow = DashboardOAuthFlow(flow_id="flow-5", server_name="asana", profile=None, hermes_home=str(tmp_path),
                              redirect_uri="https://agent.example/mcp/oauth/callback/flow-5")
    cfg = {"client_id": "pre-registered", "client_secret": "s", "redirect_host": "localhost", "redirect_port": port}
    with dashboard_oauth_flow(flow), force_interactive_oauth():
        assert _configure_callback_port(cfg, HermesTokenStorage("asana")) == port
        assert str(_build_client_metadata(cfg).redirect_uris[0]) == f"http://localhost:{port}/callback"
        asyncio.run(_make_redirect_handler(port)("https://idp.example/authorize?state=state-5"))
        assert flow.authorization_url == "https://idp.example/authorize?state=state-5"  # dashboard shows the URL

        async def _authorize():
            waiter = asyncio.ensure_future(_make_callback_waiter(port, timeout=10)())
            await asyncio.sleep(0.3)  # listener bound
            await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/callback?code=code-5&state=state-5", timeout=5).read())
            return await waiter

        result = asyncio.run(_authorize())
    assert (result.code, result.state) == ("code-5", "state-5")
