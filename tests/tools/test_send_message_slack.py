"""Slack-specific send_message delivery regressions.

Salvaged from #47547 and adapted to the post-#41112 plugin layout: the legacy
``_send_slack`` helper moved to ``plugins/platforms/slack/adapter.py::
_standalone_send`` and text sends now route through ``_send_via_adapter``
(live adapter first, registry standalone fallback).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def _ensure_slack_mock(monkeypatch):
    """Install lightweight Slack modules when optional Slack deps are absent."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_slack_send_to_platform_routes_through_send_via_adapter(monkeypatch):
    """Slack text sends go through _send_via_adapter (live adapter first)."""
    _ensure_slack_mock(monkeypatch)

    live_send = AsyncMock(return_value={"success": True, "message_id": "live-ts"})

    with patch("tools.send_message_tool._send_via_adapter", live_send):
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                SimpleNamespace(enabled=True, token="bad-token,good-token", extra={}),
                "C123",
                "**hello** from Hermes",
                thread_id="171.1",
            )
        )

    assert result == {"success": True, "message_id": "live-ts"}
    live_send.assert_awaited_once()
    call = live_send.await_args
    assert call.args[0] == Platform.SLACK
    assert call.args[2] == "C123"
    assert call.kwargs["thread_id"] == "171.1"


class _SlackResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _SlackPostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SlackSession:
    """Fake aiohttp session whose good-token posts succeed."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, json, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append((token, json))
        if token == "good-token":
            payload = {"ok": True, "ts": "171.123"}
        else:
            payload = {"ok": False, "error": "invalid_auth"}
        return _SlackPostContext(_SlackResponse(payload))


@pytest.fixture
def _standalone_send(monkeypatch):
    _ensure_slack_mock(monkeypatch)
    from plugins.platforms.slack import adapter as slack_adapter

    return slack_adapter._standalone_send


def test_standalone_send_stops_on_non_token_error(monkeypatch, _standalone_send):
    """Terminal errors (not token-scoped) must not burn the remaining tokens."""

    class _FatalSession(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            self.calls.append((token, json))
            return _SlackPostContext(
                _SlackResponse({"ok": False, "error": "msg_too_long"})
            )

    fake_session = _FatalSession()
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *args, **kwargs: fake_session
    )

    pconfig = SimpleNamespace(enabled=True, token="tok-a,tok-b", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello"))

    assert result == {"error": "Slack API error: msg_too_long"}
    assert len(fake_session.calls) == 1


def test_standalone_send_drops_non_dotted_thread_ts(monkeypatch, _standalone_send):
    """A bare numeric thread id (e.g. a migrated Telegram topic) must not be
    passed to Slack as ``thread_ts`` — Slack rejects it with
    ``invalid_thread_ts`` and the whole cron delivery fails silently (#86264).
    Drop it and post to the channel root instead."""
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: fake_session)

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        _standalone_send(pconfig, "C123", "hello", thread_id="15900")
    )

    assert "error" not in result
    assert len(fake_session.calls) == 1
    _token, body = fake_session.calls[0]
    assert "thread_ts" not in body


def test_standalone_send_keeps_valid_thread_ts(monkeypatch, _standalone_send):
    """A well-formed dotted Slack ts is preserved as the thread anchor."""
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: fake_session)

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        _standalone_send(pconfig, "C123", "hello", thread_id="1718000000.123456")
    )

    assert "error" not in result
    assert len(fake_session.calls) == 1
    _token, body = fake_session.calls[0]
    assert body.get("thread_ts") == "1718000000.123456"


def test_standalone_send_drops_unicode_digit_thread_ts(
    monkeypatch, _standalone_send
):
    """A dotted ts written with non-ASCII (Unicode) digits is not a valid Slack
    ``thread_ts``. Python's ``\\d`` would match it, so the anchor must be
    validated against ASCII digits only and dropped in favour of the channel
    root — otherwise Slack rejects it with ``invalid_thread_ts`` (#86264)."""
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: fake_session)

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        # "1718000000.123456" in Arabic-Indic digits.
        _standalone_send(
            pconfig, "C123", "hello", thread_id="١٧١٨٠٠٠٠٠٠.١٢٣٤٥٦"
        )
    )

    assert "error" not in result
    assert len(fake_session.calls) == 1
    _token, body = fake_session.calls[0]
    assert "thread_ts" not in body


class _FakeAsyncWebClient:
    """Records ``chat_postMessage`` / ``files_upload_v2`` kwargs for the media
    path so tests can assert the ``thread_ts`` that reaches Slack uploads."""

    created: list = []

    def __init__(self, token=None, **_kwargs):
        self.token = token
        self.post_calls: list = []
        self.upload_calls: list = []
        _FakeAsyncWebClient.created.append(self)

    async def chat_postMessage(self, **kwargs):
        self.post_calls.append(kwargs)
        return {"ok": True, "ts": "1718000000.999999"}

    async def files_upload_v2(self, **kwargs):
        self.upload_calls.append(kwargs)
        return {"ok": True, "file": {"timestamp": "1718000000.888888"}}


@pytest.fixture
def _fake_web_client(monkeypatch):
    """Inject a recording ``AsyncWebClient`` for ``_standalone_send``'s media
    path (function-local ``from slack_sdk.web.async_client import ...``)."""
    _ensure_slack_mock(monkeypatch)
    _FakeAsyncWebClient.created = []
    monkeypatch.setitem(
        sys.modules,
        "slack_sdk.web.async_client",
        SimpleNamespace(AsyncWebClient=_FakeAsyncWebClient),
    )
    return _FakeAsyncWebClient


def test_standalone_send_media_upload_drops_invalid_thread_ts(
    tmp_path, _standalone_send, _fake_web_client
):
    """The coerced ``thread_ts`` must also gate the ``files_upload_v2`` media
    path — a bare numeric id has to be dropped from the upload, not just from
    the text ``chat.postMessage``, or media cron sends fail with
    ``invalid_thread_ts`` (#86264)."""
    media = tmp_path / "chart.png"
    media.write_bytes(b"\x89PNG\r\n")

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        _standalone_send(
            pconfig,
            "C123",
            "",
            thread_id="15900",
            media_files=[(str(media), False)],
            caption="here you go",
        )
    )

    assert "error" not in result
    client = _fake_web_client.created[-1]
    assert len(client.upload_calls) == 1
    assert "thread_ts" not in client.upload_calls[0]
    assert client.upload_calls[0]["initial_comment"]


def test_standalone_send_media_upload_keeps_valid_thread_ts(
    tmp_path, _standalone_send, _fake_web_client
):
    """A well-formed dotted ts is preserved as the upload's ``thread_ts`` so a
    valid media reply still lands in-thread."""
    media = tmp_path / "chart.png"
    media.write_bytes(b"\x89PNG\r\n")

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        _standalone_send(
            pconfig,
            "C123",
            "",
            thread_id="1718000000.123456",
            media_files=[(str(media), False)],
            caption="here you go",
        )
    )

    assert "error" not in result
    client = _fake_web_client.created[-1]
    assert len(client.upload_calls) == 1
    assert client.upload_calls[0].get("thread_ts") == "1718000000.123456"


def test_standalone_send_caption_fallback_drops_invalid_thread_ts(
    tmp_path, _standalone_send, _fake_web_client
):
    """When the media file is missing, the caption rides a fallback
    ``chat.postMessage`` — that request must drop an invalid ``thread_ts`` too."""
    missing = tmp_path / "gone.png"

    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    result = asyncio.run(
        _standalone_send(
            pconfig,
            "C123",
            "",
            thread_id="15900",
            media_files=[(str(missing), False)],
            caption="caption text",
        )
    )

    assert "warnings" in result
    client = _fake_web_client.created[-1]
    assert client.upload_calls == []
    assert len(client.post_calls) == 1
    assert "thread_ts" not in client.post_calls[0]
