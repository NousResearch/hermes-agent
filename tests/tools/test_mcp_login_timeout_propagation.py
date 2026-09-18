"""Login-path OAuth must not be killed by the 60s handshake default.

``hermes mcp login`` (and the dashboard's re-auth) probe the server with a
connect timeout floored at 315s — the 300s OAuth callback window plus headroom
(see ``_reauth_oauth_server``). But that floor was never propagated into the
config the transport reads: ``MCPServerTask._run_http`` falls back to
``_DEFAULT_CONNECT_TIMEOUT`` (60s) for the initialize handshake, which — while
an interactive browser OAuth is parked on its callback listener — fires at 58s,
retries the connection, and starts a SECOND authorization round that collides
with the first round's still-bound listener (``EADDRINUSE``), killing the login
(#99984 bug 2; observed live against mcp.figma.com 2026-09-08).

The fix: when a probe is given an explicit ``connect_timeout``, it is written
into the resolved server config so the transport's handshake wait and the
caller's wait share one budget. These tests pin that contract.
"""
from __future__ import annotations

import asyncio

import pytest

import hermes_cli.mcp_config as mc


def test_probe_propagates_connect_timeout_into_config(monkeypatch):
    """An explicit probe timeout must reach the transport as
    ``connect_timeout`` in the server config (one budget for caller + handshake)."""
    from tools import mcp_tool as core
    from tools import mcp_tool_discovery as disc

    seen: list[dict] = []

    async def _fake_connect(name, config):
        seen.append(config)
        # Mirror MCPServerTask._run_http's read: what the transport would bound
        # the initialize handshake by for this config.
        bound = float(config.get("connect_timeout", core._DEFAULT_CONNECT_TIMEOUT))
        seen.append({"_handshake_bound": bound})
        raise asyncio.TimeoutError()

    monkeypatch.setattr(disc, "_connect_server", _fake_connect)

    with pytest.raises(Exception):
        mc._probe_single_server("srv", {"url": "https://example.com/mcp", "auth": "oauth"},
                                connect_timeout=315.0)

    bounds = [r["_handshake_bound"] for r in seen if isinstance(r, dict) and "_handshake_bound" in r]
    assert bounds, "probe must reach _connect_server"
    assert all(b >= 315.0 for b in bounds), (
        f"transport handshake must be bounded by the caller's connect_timeout, got {bounds}")
