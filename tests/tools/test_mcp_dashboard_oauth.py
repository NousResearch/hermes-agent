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


def test_mark_error_surfaces_real_cause_to_callback_waiter():
    """A failure marked before any browser redirect (worker crash,
    authorization-URL timeout, user cancel) must reach the SDK's callback
    waiter, not the generic "did not include an authorization code" line."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-err",
        server_name="asana",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-err",
    )
    flow.mark_error("403 Forbidden from the OAuth registration endpoint")

    assert flow.snapshot()["error"] == "403 Forbidden from the OAuth registration endpoint"
    with pytest.raises(RuntimeError, match="403 Forbidden from the OAuth registration endpoint"):
        asyncio.run(flow.wait_for_callback())


def test_mark_error_with_empty_message_stays_diagnosable():
    """str() of a bare TimeoutError() is ""; the waiter must still report a
    flow failure instead of falling through to the no-code message."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-empty",
        server_name="asana",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-empty",
    )
    flow.mark_error("")

    with pytest.raises(RuntimeError, match="empty error message"):
        asyncio.run(flow.wait_for_callback())


def test_late_mark_error_cannot_override_delivered_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-late",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-late",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s9"))
    flow.deliver_callback(code="code-9", state="s9", error=None)

    flow.mark_error("worker crashed after the browser redirected")
    assert asyncio.run(flow.wait_for_callback())[:2] == ("code-9", "s9")


def test_failed_reauth_rollback_preserves_newer_oauth_state(tmp_path, monkeypatch):
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("reports")
    storage._tokens_path().parent.mkdir(parents=True)
    storage._tokens_path().write_text("OLD")
    backup = storage.snapshot()
    storage.remove()

    storage._tokens_path().write_text("FRESH")
    storage.restore(backup, only_if_absent=True)

    assert storage._tokens_path().read_text() == "FRESH"
