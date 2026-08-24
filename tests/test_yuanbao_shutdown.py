"""test_yuanbao_shutdown.py - Yuanbao adapter shutdown teardown timing.

Regression coverage for #40383: a non-responsive Yuanbao WS server must not
stall gateway shutdown. ``websockets`` ``ws.close()`` blocks up to the
connection's ``close_timeout`` (5s) waiting for the server's close-frame echo;
on an idle shutdown the server never replies, so ``_cleanup_ws`` used to wait
the full ~5s. The cleanup path now bounds the close await so a hung server
cannot stall teardown.

These tests assert the *bounding/timing* contract of ``_cleanup_ws`` using
lightweight fakes; force-closing the underlying TCP transport on cancellation
is ``websockets``' responsibility (and harmless at shutdown, where the loop is
tearing down regardless), so it is intentionally out of scope here.
"""

import sys
import os
import asyncio

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from gateway.config import PlatformConfig
from gateway.platforms.yuanbao import (
    YuanbaoAdapter,
    ConnectionManager,
    WS_CLOSE_TIMEOUT_S,
)


def make_config(**kwargs):
    extra = kwargs.pop("extra", {})
    extra.setdefault("app_id", "test_key")
    extra.setdefault("app_secret", "test_secret")
    extra.setdefault("ws_url", "wss://test.example.com/ws")
    extra.setdefault("api_domain", "https://test.example.com")
    return PlatformConfig(extra=extra, **kwargs)


class _HangingWS:
    """Fake WS whose close() never gets a server echo — sleeps past the bound."""

    def __init__(self, sleep_s: float):
        self._sleep_s = sleep_s
        self.close_called = False

    async def close(self):
        self.close_called = True
        await asyncio.sleep(self._sleep_s)


class _FastWS:
    """Fake WS whose close() returns promptly (responsive server)."""

    def __init__(self):
        self.close_called = False

    async def close(self):
        self.close_called = True


class _RaisingWS:
    async def close(self):
        raise RuntimeError("connection already reset")


def _connection() -> ConnectionManager:
    return YuanbaoAdapter(make_config())._connection


@pytest.mark.asyncio
async def test_cleanup_ws_does_not_stall_on_hung_server():
    """A server that never echoes the close frame must not stall teardown."""
    cm = _connection()
    hung = _HangingWS(sleep_s=WS_CLOSE_TIMEOUT_S + 4.0)
    cm._ws = hung

    loop = asyncio.get_running_loop()
    start = loop.time()
    await cm._cleanup_ws()
    elapsed = loop.time() - start

    assert hung.close_called
    assert cm._ws is None
    # Bounded by WS_CLOSE_TIMEOUT_S (+ small scheduling slack), not the 5s
    # close_timeout the server would otherwise hold us to.
    assert elapsed < WS_CLOSE_TIMEOUT_S + 1.0


@pytest.mark.asyncio
async def test_cleanup_ws_fast_path_returns_immediately():
    """A responsive server completes the handshake well under the bound."""
    cm = _connection()
    fast = _FastWS()
    cm._ws = fast

    loop = asyncio.get_running_loop()
    start = loop.time()
    await cm._cleanup_ws()
    elapsed = loop.time() - start

    assert fast.close_called
    assert cm._ws is None
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_cleanup_ws_swallows_close_errors():
    """A close() that raises must still clear the ws reference."""
    cm = _connection()
    cm._ws = _RaisingWS()

    await cm._cleanup_ws()

    assert cm._ws is None
