"""test_yuanbao_reconnect_set_active.py - Verify _do_reconnect restores the active singleton.

Regression test for #58363: after a WS disconnect/reconnect cycle,
``get_active_adapter()`` must return the live adapter (not ``None``).
The original ``_do_reconnect()`` succeeded but never called
``YuanbaoAdapter.set_active()``, leaving the singleton permanently
``None`` until a full gateway restart.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from gateway.platforms.yuanbao import (
    YuanbaoAdapter,
    ConnectionManager,
    get_active_adapter,
)


def _make_adapter(**kwargs):
    """Create a minimal YuanbaoAdapter mock."""
    adapter = MagicMock(spec=YuanbaoAdapter)
    adapter.name = "yuanbao"
    adapter._app_key = "test_key"
    adapter._app_secret = "test_secret"
    adapter._api_domain = "https://test.example.com"
    adapter._route_env = None
    adapter._bot_id = "test_bot"
    adapter._ws_url = "wss://test.example.com/ws"
    adapter._mark_connected = MagicMock()
    adapter._mark_disconnected = MagicMock()
    adapter._release_platform_lock = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_do_reconnect_calls_set_active_on_success():
    """After a successful reconnect, set_active(adapter) must be called."""
    adapter = _make_adapter()
    cm = ConnectionManager(adapter)

    # Mock the reconnect internals to succeed on first attempt
    mock_ws = AsyncMock()
    mock_ws.close = AsyncMock()

    with (
        patch.object(cm, "_cleanup_ws", new_callable=AsyncMock) as mock_cleanup,
        patch(
            "gateway.platforms.yuanbao.SignManager.force_refresh",
            new_callable=AsyncMock,
            return_value={"bot_id": "test_bot", "token": "test_token"},
        ),
        patch("gateway.platforms.yuanbao.websockets.connect", new_callable=AsyncMock, return_value=mock_ws),
        patch.object(cm, "_authenticate", new_callable=AsyncMock, return_value=True),
        patch.object(cm, "_heartbeat_loop", new_callable=AsyncMock),
        patch.object(cm, "_receive_loop", new_callable=AsyncMock),
        patch("gateway.platforms.yuanbao.MAX_RECONNECT_ATTEMPTS", 1),
    ):
        # Clear any existing active instance
        YuanbaoAdapter.set_active(None)
        assert get_active_adapter() is None

        # Run reconnect
        result = await cm._do_reconnect()

        # Reconnect should succeed
        assert result is True

        # After successful reconnect, get_active() must return the adapter
        assert get_active_adapter() is adapter


@pytest.mark.asyncio
async def test_do_reconnect_does_not_set_active_on_failure():
    """When all reconnect attempts fail, set_active should NOT be called."""
    adapter = _make_adapter()
    cm = ConnectionManager(adapter)

    with (
        patch.object(cm, "_cleanup_ws", new_callable=AsyncMock),
        patch(
            "gateway.platforms.yuanbao.SignManager.force_refresh",
            new_callable=AsyncMock,
            side_effect=Exception("auth failed"),
        ),
        patch("gateway.platforms.yuanbao.MAX_RECONNECT_ATTEMPTS", 1),
    ):
        # Clear any existing active instance
        YuanbaoAdapter.set_active(None)
        assert get_active_adapter() is None

        # Run reconnect - should fail
        result = await cm._do_reconnect()

        # Reconnect should fail
        assert result is False

        # get_active() should still be None
        assert get_active_adapter() is None
