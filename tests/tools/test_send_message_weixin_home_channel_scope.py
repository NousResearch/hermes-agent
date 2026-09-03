"""Regression test for the WEIXIN_HOME_CHANNEL multiplex secret-scope leak.

``_handle_send``'s weixin home-channel fallback (used when weixin is
configured purely via ``.env``, with no ``gateway.yaml``/``config.yaml``
home channel) read ``WEIXIN_HOME_CHANNEL`` through raw ``os.getenv`` instead
of ``get_secret`` -- unlike the WEIXIN_TOKEN/ACCOUNT_ID/BASE_URL/CDN_BASE_URL
resolution a few lines above it in the same function, and unlike
``cron/scheduler.py``'s home-target chat-id resolution, which was fixed to
read through ``get_secret`` for the same reason. Under multiplexing, a
secondary profile's ``send_message`` call would resolve the destination
chat from the shared process's ``os.environ`` (the default profile's value)
instead of its own ``.env``-scoped ``WEIXIN_HOME_CHANNEL``.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent import secret_scope
from gateway.config import Platform
from tools.send_message_tool import send_message_tool


@pytest.fixture()
def multiplex_on():
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous)


def _weixin_config():
    # No gateway.yaml entry and no configured home channel -- weixin's
    # "purely via .env" path, which is what reaches the fallback under test.
    return SimpleNamespace(platforms={}, get_home_channel=lambda _platform: None)


def _run_async_immediately(coro):
    return asyncio.run(coro)


class TestWeixinHomeChannelSecretScope:
    def test_scoped_home_channel_wins_over_environ(self, multiplex_on, monkeypatch):
        """A secondary profile's own WEIXIN_HOME_CHANNEL must win over the
        shared process os.environ's (default profile's) value."""
        monkeypatch.setenv("WEIXIN_TOKEN", "default-token")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "default-account")
        monkeypatch.setenv("WEIXIN_HOME_CHANNEL", "default-profile-chat")

        token = secret_scope.set_secret_scope({
            "WEIXIN_TOKEN": "scoped-token",
            "WEIXIN_ACCOUNT_ID": "scoped-account",
            "WEIXIN_HOME_CHANNEL": "scoped-profile-chat",
        })
        try:
            with patch("gateway.config.load_gateway_config", return_value=_weixin_config()), \
                 patch("tools.interrupt.is_interrupted", return_value=False), \
                 patch("model_tools._run_async", side_effect=_run_async_immediately), \
                 patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
                 patch("gateway.mirror.mirror_to_session", return_value=True):
                result = json.loads(
                    send_message_tool({"action": "send", "target": "weixin", "message": "hi"})
                )
        finally:
            secret_scope.reset_secret_scope(token)

        assert result["success"] is True
        send_mock.assert_awaited_once()
        assert send_mock.await_args.args[0] == Platform.WEIXIN
        assert send_mock.await_args.args[2] == "scoped-profile-chat"

    def test_scoped_miss_does_not_borrow_environ_home_channel(self, multiplex_on, monkeypatch):
        """A secondary profile with no WEIXIN_HOME_CHANNEL of its own must
        fail closed, not send to the default profile's home chat."""
        monkeypatch.setenv("WEIXIN_TOKEN", "default-token")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "default-account")
        monkeypatch.setenv("WEIXIN_HOME_CHANNEL", "default-profile-chat")

        token = secret_scope.set_secret_scope({
            "WEIXIN_TOKEN": "scoped-token",
            "WEIXIN_ACCOUNT_ID": "scoped-account",
        })
        try:
            with patch("gateway.config.load_gateway_config", return_value=_weixin_config()), \
                 patch("tools.interrupt.is_interrupted", return_value=False), \
                 patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
                result = json.loads(
                    send_message_tool({"action": "send", "target": "weixin", "message": "hi"})
                )
        finally:
            secret_scope.reset_secret_scope(token)

        assert "error" in result
        assert "WEIXIN_HOME_CHANNEL" in result["error"]
        send_mock.assert_not_awaited()

    def test_single_profile_deployment_falls_back_to_environ(self, monkeypatch):
        """Outside multiplexing (the common single-profile deployment, no
        ``multiplex_on``/no scope installed) ``get_secret`` still falls
        through to ``os.environ`` -- the fix must not regress this path."""
        monkeypatch.setenv("WEIXIN_TOKEN", "default-token")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "default-account")
        monkeypatch.setenv("WEIXIN_HOME_CHANNEL", "default-profile-chat")

        with patch("gateway.config.load_gateway_config", return_value=_weixin_config()), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool({"action": "send", "target": "weixin", "message": "hi"})
            )

        assert result["success"] is True
        assert send_mock.await_args.args[2] == "default-profile-chat"
