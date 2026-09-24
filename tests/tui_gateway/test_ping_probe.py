"""Tests for the ``ping`` liveness probe (tui_gateway/server.py).

The desktop client probes ``ping`` after sleep/wake to distinguish a
half-open TCP socket (connectionState still ``open`` while every RPC hangs)
from a genuinely healthy connection. The contract is minimal on purpose:
answered synchronously on the WS reader thread, no session, no IO, no agent.
"""

from __future__ import annotations

import tui_gateway.server as srv


def _call(method: str, params: dict) -> dict:
    """Invoke a registered RPC method and return its result dict."""
    envelope = srv._methods[method](1, params)
    return envelope["result"]


def test_ping_registered_and_answers_pong():
    res = _call("ping", {})
    assert res == {"pong": True}


