"""QQ Bot proactive text send honors markdown_support (#26697)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.platforms.qqbot.constants import MSG_TYPE_MARKDOWN, MSG_TYPE_TEXT
from tools.send_message_tool import (
    _qqbot_build_text_payload,
    _qqbot_markdown_support,
    _qqbot_send_text_message,
    _send_qqbot,
)


def _pconfig(*, markdown_support=True):
    extra = {"app_id": "app-123", "client_secret": "client-secret"}
    extra["markdown_support"] = markdown_support
    return SimpleNamespace(token="client-secret", extra=extra, enabled=True)


class TestQqbotMarkdownSupportFlag:
    def test_default_true_when_extra_omits_key(self):
        assert _qqbot_markdown_support(SimpleNamespace(extra={})) is True

    def test_explicit_false(self):
        assert _qqbot_markdown_support(_pconfig(markdown_support=False)) is False


class TestQqbotBuildTextPayload:
    def test_markdown_matches_adapter_shape(self):
        table = "| Name | Qty |\n| --- | --- |\n| Apple | 3 |"
        body = _qqbot_build_text_payload(table, markdown_support=True)
        assert body["msg_type"] == MSG_TYPE_MARKDOWN
        assert body["markdown"]["content"] == table
        assert "content" not in body

    def test_plain_text_when_disabled(self):
        body = _qqbot_build_text_payload("**bold**", markdown_support=False)
        assert body["msg_type"] == MSG_TYPE_TEXT
        assert body["content"] == "**bold**"
        assert "markdown" not in body

    def test_guild_is_content_only_even_when_markdown_on(self):
        body = _qqbot_build_text_payload("hi", markdown_support=True, guild=True)
        assert body == {"content": "hi"}


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class TestQqbotSendTextMessagePayload:
    def test_c2c_uses_markdown_body_after_channel_miss(self):
        posts = []

        class _Client:
            async def post(self, url, json=None, headers=None):
                posts.append((url, json))
                if "/channels/" in url:
                    return _Resp(404)
                if "/v2/users/" in url:
                    return _Resp(200, {"id": "m-c2c"})
                return _Resp(500)

        result = asyncio.run(
            _qqbot_send_text_message(
                _Client(),
                {},
                "openid-1",
                "| A | B |\n| --- | --- |\n| 1 | 2 |",
                markdown_support=True,
            )
        )

        assert result["success"] is True
        assert result["message_id"] == "m-c2c"
        channel_url, channel_body = posts[0]
        c2c_url, c2c_body = posts[1]
        assert "/channels/" in channel_url
        assert channel_body == {"content": "| A | B |\n| --- | --- |\n| 1 | 2 |"}
        assert "/v2/users/openid-1/messages" in c2c_url
        assert c2c_body["msg_type"] == MSG_TYPE_MARKDOWN
        assert c2c_body["markdown"]["content"].startswith("| A | B |")

    def test_c2c_uses_plain_body_when_markdown_disabled(self):
        posts = []

        class _Client:
            async def post(self, url, json=None, headers=None):
                posts.append(json)
                if "/channels/" in url:
                    return _Resp(404)
                if "/v2/users/" in url:
                    return _Resp(200, {"id": "m-plain"})
                return _Resp(500)

        result = asyncio.run(
            _qqbot_send_text_message(
                _Client(),
                {},
                "openid-1",
                "**bold**",
                markdown_support=False,
            )
        )

        assert result["success"] is True
        c2c_body = posts[1]
        assert c2c_body["msg_type"] == MSG_TYPE_TEXT
        assert c2c_body["content"] == "**bold**"
        assert "markdown" not in c2c_body


class TestSendQqbotHonorsConfig:
    def test_text_send_forwards_markdown_flag(self):
        token_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"access_token": "at-1"},
        )

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None, headers=None):
                return token_resp

        class _HttpxMod:
            AsyncClient = _Client

        text_send = AsyncMock(
            return_value={"success": True, "platform": "qqbot", "chat_id": "oid"}
        )
        with patch.dict("sys.modules", {"httpx": _HttpxMod()}), patch(
            "tools.send_message_tool._qqbot_send_text_message",
            new=text_send,
        ):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig(markdown_support=False),
                    "oid",
                    "| Name | Qty |",
                )
            )

        assert result["success"] is True
        assert text_send.await_args.kwargs["markdown_support"] is False
        assert text_send.await_args.args[3] == "| Name | Qty |"
