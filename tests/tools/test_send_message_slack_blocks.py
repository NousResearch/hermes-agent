"""Out-of-process Slack sends (``hermes send``, cron delivery) honour the same Block Kit opt-ins
(``rich_blocks`` / ``markdown_blocks``) as gateway replies, so a markdown table renders as a native
Slack table instead of a fenced code block."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.tools.test_send_message_slack import (
    _ensure_slack_mock, _SlackPostContext, _SlackResponse, _SlackSession)

TABLE_MESSAGE = "Status\n\n| item | value |\n|---|---|\n| a | 1 |\n| b | 2 |\n"


@pytest.fixture
def standalone_send(monkeypatch):
    _ensure_slack_mock(monkeypatch)
    from plugins.platforms.slack import adapter as slack_adapter

    return slack_adapter._standalone_send


def _send(monkeypatch, standalone_send, session, extra):
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra=extra)
    return asyncio.run(standalone_send(pconfig, "C123", TABLE_MESSAGE, thread_id="171.000"))


def test_rich_blocks_table_sent_as_native_table_block(monkeypatch, standalone_send):
    session = _SlackSession()
    result = _send(monkeypatch, standalone_send, session, {"rich_blocks": True})

    assert result["success"] is True
    payload = session.calls[-1][1]
    assert payload["thread_ts"] == "171.000"
    assert any(block["type"] == "table" for block in payload["blocks"])
    assert payload["text"]  # notification / fallback text still present


def test_blocks_absent_without_opt_in(monkeypatch, standalone_send):
    session = _SlackSession()
    _send(monkeypatch, standalone_send, session, {})

    assert "blocks" not in session.calls[-1][1]


def test_rejected_blocks_retry_as_text(monkeypatch, standalone_send):
    class _RejectBlocksSession(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            self.calls.append((headers["Authorization"], dict(json)))
            if "blocks" in json:
                return _SlackPostContext(_SlackResponse({"ok": False, "error": "invalid_blocks"}))
            return _SlackPostContext(_SlackResponse({"ok": True, "ts": "171.123"}))

    session = _RejectBlocksSession()
    result = _send(monkeypatch, standalone_send, session, {"rich_blocks": True})

    assert result == {"success": True, "platform": "slack", "chat_id": "C123",
                      "message_id": "171.123"}
    assert ["blocks" in call[1] for call in session.calls] == [True, False]
