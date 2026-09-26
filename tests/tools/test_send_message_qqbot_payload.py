"""``_send_qqbot`` (standalone REST path) must send the payload shape QQ renders.

Regression for #26697. Cron delivery prefers the live gateway adapter and
falls through to ``tools.send_message_senders._send_qqbot`` when that adapter
cannot be used in time (contended event loop, no adapter, restart). The
standalone sender hardcoded ``msg_type: 0`` with a top-level ``content``, so
markdown, and inline ``$...$`` math, arrived as raw source on exactly the
fallback lane — while the same message through the live adapter rendered.

The reference shape is the adapter's: ``QQAdapter._build_text_body`` sends
``{"markdown": {"content": ...}, "msg_type": 2}`` for C2C/group when
``markdown_support`` is on (default), and ``QQAdapter._send_guild_text`` sends a
bare ``{"content": ...}`` to the guild channel endpoint, which has no
``msg_type`` at all. Both are pinned here.

No QQ account, no network: ``httpx`` is faked.
"""

import sys
import types

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot.constants import MSG_TYPE_MARKDOWN, MSG_TYPE_TEXT

TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
CHANNEL_URL = "https://api.sgroup.qq.com/channels/{chat}/messages"
CHAT = "openid-1"
CONTENT = "**bold** and $a\\leqslant b$"


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload

    def json(self):
        return self._payload


def _install_fake_httpx(monkeypatch, statuses):
    """Route ``import httpx`` to a recorder over ``statuses`` (URL template -> status),
    returning the live POST list (poll via ``_message_posts``; the first is the token)."""
    calls = []

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            calls.append((url, kwargs.get("json") or {}))
            if url == TOKEN_URL:
                return _Response(200, {"access_token": "tok"})
            for template, status in statuses.items():
                if url == template.format(chat=CHAT):
                    return _Response(status, {"id": "msg-1"} if status in {200, 201} else {})
            return _Response(404, {})

    module = types.ModuleType("httpx")
    module.AsyncClient = _AsyncClient
    monkeypatch.setitem(sys.modules, "httpx", module)
    return calls


def _message_posts(calls):
    """The POSTs carrying the message itself (skip the token request)."""
    return [c for c in calls if c[0] != TOKEN_URL]


async def _send(message=CONTENT, **extra):
    from tools.send_message_tool import _send_qqbot

    pconfig = PlatformConfig(
        enabled=True, extra={"app_id": "app", "client_secret": "sec", **extra})
    return await _send_qqbot(pconfig, CHAT, message)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["v2/users", "v2/groups"])
async def test_c2c_and_group_send_markdown_by_default(monkeypatch, endpoint):
    url = f"https://api.sgroup.qq.com/{endpoint}/{{chat}}/messages"
    calls = _install_fake_httpx(monkeypatch, {CHANNEL_URL: 404, url: 200})

    result = await _send()

    assert result.get("success") is True and result.get("message_id") == "msg-1"
    assert _message_posts(calls)[-1] == (url.format(chat=CHAT),
                         {"markdown": {"content": CONTENT}, "msg_type": MSG_TYPE_MARKDOWN})


@pytest.mark.asyncio
async def test_guild_channel_endpoint_keeps_the_plain_content_body(monkeypatch):
    """The channel endpoint is a different API: a markdown body carries no
    ``content`` field there, so a 2xx would deliver nothing."""
    calls = _install_fake_httpx(monkeypatch, {CHANNEL_URL: 200})

    result = await _send()

    assert result.get("success") is True
    assert _message_posts(calls) == [(CHANNEL_URL.format(chat=CHAT),
                                      {"content": CONTENT, "msg_type": MSG_TYPE_TEXT})]


@pytest.mark.asyncio
async def test_markdown_support_false_sends_plain_text(monkeypatch):
    """The config gate the adapter honors must gate the standalone path too."""
    url = f"https://api.sgroup.qq.com/v2/users/{{chat}}/messages"
    calls = _install_fake_httpx(monkeypatch, {CHANNEL_URL: 404, url: 200})

    result = await _send(markdown_support=False)

    assert result.get("success") is True
    assert _message_posts(calls)[-1] == (url.format(chat=CHAT),
                                         {"content": CONTENT, "msg_type": MSG_TYPE_TEXT})
