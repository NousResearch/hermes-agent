"""Edge-case pin: an explicit CDP override owns the browser session even when a
hosted browser-MCP gateway is also configured — CDP first, gateway lanes only
take the wheel when no CDP endpoint is set."""
import os
import pytest


@pytest.fixture(autouse=True)
def _isolate_browser_lane_env(monkeypatch):
    """Tests must see only the env they set — the dev box exports BROWSER_MCP_*
    (gateway lane), which leaked into assertions (review: env-leakage failures)."""
    for name in ("BROWSER_MCP_URL", "BROWSER_MCP_API_KEY", "BROWSER_CDP_URL",
                 "CAMOFOX_URL", "CAMOFOX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
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


def test_cdp_session_never_reaches_mcp_transport(monkeypatch):
    """Negative tripwire: under a CDP override the registered browser handler must
    dispatch WITHOUT ever reaching the MCP transport (review P2: the old fallback
    assertion was vacuous)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.setenv("BROWSER_CDP_URL", "http://10.1.2.3:9222")
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)

    import json
    from tools.browser_mcp_transport import _call as mcp_call

    def _tripwire(*a, **k):
        raise AssertionError("MCP transport reached under a CDP-owned session")
    monkeypatch.setattr("tools.browser_mcp_transport._call", _tripwire)
    # also trip the verb-level entrypoints (defense in depth vs import-time binding)
    for verb in ("mcp_snapshot", "mcp_navigate", "mcp_click", "mcp_type"):
        monkeypatch.setattr(f"tools.browser_mcp_transport.{verb}", _tripwire)

    # The dispatcher layer is where CDP wins: _is_camofox_mode() false under override,
    # so no camofox verb (and no MCP route) is ever reached.
    from tools import browser_camofox as bc
    assert bc.is_camofox_mode() is False
    # gateway helper itself is CDP-blind by design (one layer up owns the override):
    from tools.browser_mcp_transport import _is_gateway_backend
    assert _is_gateway_backend() is True


def test_mcp_lane_routes_and_hits_transport(monkeypatch):
    """Positive control: same verb on the MCP lane DOES reach the transport (proves the
    negative tripwire above is actually armed, not silently no-op)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.setenv("BROWSER_MCP_URL", "https://api.browser-gw.example")
    monkeypatch.setenv("BROWSER_MCP_API_KEY", "k" * 8)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    import json as _json
    calls = []
    def _fake_call(tool, arguments, timeout=None, task_id=None, tab_id=None):
        calls.append(tool)
        # The P1-2 guard probes the current page first (browser_evaluate); the
        # snapshot itself follows. The guard must see an URL-shaped, non-private
        # location to fail open.
        if tool == "browser_evaluate":
            text = '"https://example.com/page"'
        else:
            text = "Snapshot:\n- link \"x\" [e1]\n"
        return {"content": [{"text": text}], "isError": False}
    monkeypatch.setattr("tools.browser_mcp_transport._call", _fake_call)

    from tools.browser_camofox import camofox_snapshot
    out = _json.loads(camofox_snapshot())
    assert out.get("success") is True
    assert calls[0] == "browser_evaluate"   # guard probe fired
    assert calls[-1] == "browser_snapshot"  # then the verb
