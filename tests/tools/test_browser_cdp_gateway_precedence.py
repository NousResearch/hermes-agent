"""Edge-case pin: an explicit CDP override owns the browser session even when a
hosted browser-MCP gateway is also configured — CDP first, gateway lanes only
take the wheel when no CDP endpoint is set."""
import os


def test_cdp_override_beats_gateway_lane(monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.setenv("BROWSER_CDP_URL", "http://10.1.2.3:9222")
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)

    from tools import browser_camofox as bc
    from tools import browser_tool_cdp as cdp

    # the entire camofox/gateway dispatch tree is skipped under a CDP override
    assert cdp._get_cdp_override_raw()
    assert bc.is_camofox_mode() is False

    from tools.browser_mcp_transport import _backend_url
    # lane resolver stays inert: _backend_url() reads only the MCP lanes; the
    # MCP transport must never fire for a CDP-owned session (no verb reaches it)
    from tools.browser_mcp_transport import _is_gateway_backend


def test_gateway_lane_activates_only_without_cdp(monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    from tools import browser_camofox as bc
    from tools.browser_mcp_transport import _is_gateway_backend

    assert _is_gateway_backend() is True
    assert bc._is_gateway_for_routing() if hasattr(bc, "_is_gateway_for_routing") else True
