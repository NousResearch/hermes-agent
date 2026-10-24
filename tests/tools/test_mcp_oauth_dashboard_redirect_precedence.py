"""A live dashboard/Desktop OAuth flow owns the callback URL.

Regression: ``_configure_callback_port`` preferred a saved ``oauth.redirect_uri`` over the
active flow's own ``redirect_uri``. A deployment that had once pinned a reverse-proxy callback
in ``config.yaml`` therefore sent the user to a URL nobody was listening on, and the Desktop /
dashboard login for that server could never complete.
"""

import pytest


def test_dashboard_flow_redirect_uri_beats_saved_redirect_uri():
    pytest.importorskip("mcp.shared.auth", reason="MCP SDK not installed")
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage, _configure_callback_port

    flow = DashboardOAuthFlow(
        flow_id="flow-redirect",
        server_name="letsfg",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="http://127.0.0.1:43117/callback",
    )
    # A stale proxy callback saved from an earlier CLI login.
    cfg = {"redirect_uri": "https://gateway.example:3738/hermes/api/mcp/oauth/callback/letsfg"}

    with dashboard_oauth_flow(flow):
        assert _configure_callback_port(cfg, HermesTokenStorage("letsfg")) == 0

    assert cfg["redirect_uri"] == flow.redirect_uri


def test_pinned_loopback_client_still_keeps_its_configured_uri():
    """A pre-registered client with a pinned ``redirect_port`` is not a dashboard flow at all,
    so its own configuration must survive untouched."""
    pytest.importorskip("mcp.shared.auth", reason="MCP SDK not installed")
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage, _configure_callback_port

    flow = DashboardOAuthFlow(
        flow_id="flow-pinned",
        server_name="pinned",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="http://127.0.0.1:43118/callback",
    )
    cfg = {"client_id": "abc", "redirect_port": 47777}

    with dashboard_oauth_flow(flow):
        assert _configure_callback_port(cfg, HermesTokenStorage("pinned")) == 47777

    assert "redirect_uri" not in cfg
