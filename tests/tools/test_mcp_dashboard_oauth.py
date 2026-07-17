"""Hosted-dashboard bridge for MCP OAuth browser callbacks."""

import asyncio

import pytest


def test_dashboard_flow_exposes_authorization_url_and_accepts_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-1",
        server_name="reports",
        profile=None,
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
    assert asyncio.run(flow.wait_for_callback()) == ("code-1", "s1")


def test_dashboard_flow_rejects_wrong_state_without_consuming_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-state",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-state",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected-state"
        )
    )

    with pytest.raises(ValueError, match="state mismatch"):
        flow.deliver_callback(code="attacker", state="wrong-state", error=None)

    flow.deliver_callback(code="legitimate", state="expected-state", error=None)
    assert asyncio.run(flow.wait_for_callback()) == (
        "legitimate",
        "expected-state",
    )


def test_dashboard_flow_rejects_second_callback():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-2",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-2",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=state"
        )
    )
    flow.deliver_callback(code="first", state="state", error=None)
    with pytest.raises(ValueError, match="already received"):
        flow.deliver_callback(code="second", state="state", error=None)


def test_dashboard_flow_cannot_resurrect_after_terminal_error():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-terminal",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-terminal",
    )
    flow.mark_error("start timed out")

    with pytest.raises(RuntimeError, match="already ended"):
        asyncio.run(
            flow.publish_authorization_url(
                "https://idp.example/authorize?state=too-late"
            )
        )

    assert flow.status == "error"
    assert flow.authorization_url is None


def test_dashboard_context_overrides_redirect_and_handlers():
    from tools.mcp_dashboard_oauth import (
        DashboardOAuthFlow,
        dashboard_oauth_flow,
        get_dashboard_oauth_flow,
    )

    flow = DashboardOAuthFlow(
        flow_id="flow-3",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/mcp/oauth/callback/flow-3",
    )
    assert get_dashboard_oauth_flow() is None
    with dashboard_oauth_flow(flow):
        assert get_dashboard_oauth_flow() is flow
    assert get_dashboard_oauth_flow() is None


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
        assert asyncio.run(_make_callback_waiter(0)()) == ("code-4", "state-4")

    assert flow.authorization_url == "https://idp.example/authorize?state=state-4"


def test_manager_build_allows_dashboard_flow_without_tty(tmp_path, monkeypatch):
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin.isatty", lambda: False)
    flow = DashboardOAuthFlow(
        flow_id="flow-5",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-5",
    )
    with dashboard_oauth_flow(flow):
        provider = MCPOAuthManager().get_or_build_provider(
            "reports", "https://mcp.example/mcp", {}
        )
    assert provider is not None
    assert str(provider.context.client_metadata.redirect_uris[0]) == flow.redirect_uri


def test_manager_evict_preserves_persisted_oauth_state(tmp_path, monkeypatch):
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("reports")
    storage._tokens_path().parent.mkdir(parents=True)
    storage._tokens_path().write_text(
        '{"access_token":"a","token_type":"Bearer"}'
    )
    manager = MCPOAuthManager()
    manager._entries["reports"] = _ProviderEntry(
        server_url="https://mcp.example/mcp", oauth_config={}
    )

    manager.evict("reports")

    assert "reports" not in manager._entries
    assert storage._tokens_path().exists()
