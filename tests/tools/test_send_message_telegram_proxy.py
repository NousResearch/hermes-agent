"""Regression tests for the standalone Telegram send path's proxy support.

The ``send_message`` tool, when invoked from a process *other than* the
gateway (agent / TUI / cron), runs ``_send_telegram`` directly instead of
delegating to the in-process gateway adapter. Before the fix that
accompanies these tests, that standalone path constructed
``telegram.Bot(token=...)`` with no proxy, so in regions where
api.telegram.org is blocked (e.g. RU) the send would just time out with
``Telegram send failed: Timed out`` and never show up in ``gateway.log``.

These tests verify that the standalone path now honours ``TELEGRAM_PROXY``
the same way the gateway adapter (and the Discord standalone path) do.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock_with_request(
    monkeypatch: pytest.MonkeyPatch,
    bot_factory: MagicMock,
    httpx_request_factory: MagicMock,
) -> None:
    """Install a stub ``telegram`` package whose ``Bot`` and
    ``telegram.request.HTTPXRequest`` are the supplied mocks.

    Mirrors ``_install_telegram_mock`` in test_send_message_tool.py but also
    provides the ``telegram.request`` submodule that the proxy branch needs.
    """
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory)
    # MessageEntity needed by #27865 mention-detection path.
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=_MessageEntity,
        constants=constants_mod,
        request=request_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "telegram.request", request_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    return bot


@pytest.mark.parametrize("proxy_url", [None, "socks5://127.0.0.1:1080"])
def test_standalone_transport_preserves_proxy_and_media_timeout(monkeypatch, proxy_url):
    from tools.send_message_senders import _send_telegram

    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
        "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    if proxy_url:
        monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

    bot = _make_bot()
    bot_factory = MagicMock(return_value=bot)
    request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
    _install_telegram_mock_with_request(monkeypatch, bot_factory, request_factory)

    result = asyncio.run(_send_telegram("tok", "123", "hello world"))

    assert result["success"] is True
    bot_factory.assert_called_once()
    kwargs = bot_factory.call_args.kwargs
    assert kwargs["token"] == "tok"
    assert "request" in kwargs and "get_updates_request" in kwargs
    assert request_factory.call_count == 2
    for call in request_factory.call_args_list:
        assert call.kwargs.get("proxy") == proxy_url
        assert call.kwargs["media_write_timeout"] == 60.0
    bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_ptb_upload_carries_media_timeout_to_httpx(tmp_path, monkeypatch):
    pytest.importorskip("telegram")
    import httpx

    from tools.send_message_senders import _send_telegram

    monkeypatch.setattr("gateway.platforms.base.resolve_proxy_url", lambda *a, **kw: None)
    payload = b"standalone upload bytes"
    document = tmp_path / "upload.txt"
    document.write_bytes(payload)
    observed = []
    clients = []
    client_type = httpx.AsyncClient

    async def receive(request):
        observed.append((request.url.path, request.extensions["timeout"], await request.aread()))
        return httpx.Response(200, json={
            "ok": True,
            "result": {"message_id": 42, "date": 0, "chat": {"id": 123, "type": "private"}},
        })

    def make_client(*args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(receive), trust_env=False)
        client = client_type(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    try:
        result = await _send_telegram(
            "123456:synthetic-test-token", "123", "", media_files=[(str(document), False)],
        )
    finally:
        for client in clients:
            await client.aclose()

    assert result["success"] is True
    assert len(observed) == 1
    path, timeout, body = observed[0]
    assert path.endswith("/sendDocument")
    assert timeout["write"] == 60.0
    assert payload in body
