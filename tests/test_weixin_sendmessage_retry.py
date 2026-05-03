"""Regression tests for Weixin sendmessage degraded retry behaviour."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.platforms import weixin


class _DummyTokenStore:
    def __init__(self, account_id: str, chat_id: str, token: str) -> None:
        self.account_id = account_id
        self.chat_id = chat_id
        self.token = token
        self._cache = {self._key(account_id, chat_id): token}
        self.cleared_keys: list[str] = []

    def _key(self, account_id: str, user_id: str) -> str:
        return f"{account_id}:{user_id}"


@pytest.mark.asyncio
async def test_send_text_chunk_retries_without_context_token_on_ilink_unknown_error(monkeypatch):
    """ret=-2 from iLink can mean a stale/invalid context token; retry tokenless once."""

    calls = []

    async def fake_send_message(session, *, base_url, token, to, text, context_token, client_id):
        calls.append(context_token)
        if len(calls) == 1:
            return {"ret": -2, "errmsg": "unknown error"}
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_message", fake_send_message)

    adapter = SimpleNamespace(
        _send_chunk_retries=2,
        _send_chunk_retry_delay_seconds=0,
        _send_session=object(),
        _base_url="https://ilinkai.weixin.qq.com",
        _token="dummy",
        _account_id="account@im.bot",
        _token_store=_DummyTokenStore("account@im.bot", "peer@im.wechat", "stale"),
        name="Weixin",
    )

    await weixin.WeixinAdapter._send_text_chunk(
        adapter,
        chat_id="peer@im.wechat",
        chunk="hello",
        context_token="stale",
        client_id="client-1",
    )
    assert calls == ["stale", None]
    assert adapter._token_store._cache == {}
