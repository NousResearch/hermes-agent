"""Tests for the shared OneBot v11 client.

Exercises real logic — URL/token configuration, availability gating, and
both the HTTP and WebSocket transports' success / failure / error paths —
with mocked HTTP and a fake WebSocket server so no network is touched.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from tools.onebot_client import (
    onebot_access_token,
    onebot_base_url,
    onebot_call,
    onebot_configured,
)


# ---------------------------------------------------------------------------
# Fake HTTP response
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Fake WebSocket server
# ---------------------------------------------------------------------------

class _FakeWS:
    """A fake OneBot WebSocket connection for _ws_roundtrip tests.

    Replies to the action request with our echo, optionally preceded by a
    burst of pushed events (no echo) to exercise the skip-until-echo loop.
    """

    def __init__(self, data=None, status="ok", retcode=0, message=None,
                 lead_events=0):
        self._data = data
        self._status = status
        self._retcode = retcode
        self._message = message
        self._lead_events = lead_events
        self.sent = []
        self._answered = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        if self._lead_events > 0:
            self._lead_events -= 1
            return json.dumps(
                {"post_type": "meta_event", "meta_event_type": "heartbeat"}
            )
        if self._answered:  # the round-trip should have returned by now
            await asyncio.sleep(3600)
        self._answered = True
        echo = json.loads(self.sent[-1])["echo"]
        reply = {"status": self._status, "retcode": self._retcode, "echo": echo}
        if self._message is not None:
            reply["message"] = self._message
        if self._status != "failed" and self._data is not None:
            reply["data"] = self._data
        return json.dumps(reply)


def _fake_connect(fake_ws):
    """Build a websockets.connect replacement that yields *fake_ws*."""

    def _connect(uri, **kwargs):
        _connect.last_uri = uri
        return fake_ws

    _connect.last_uri = None
    return _connect


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfig:
    def test_base_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000/")
        assert onebot_base_url() == "http://127.0.0.1:3000"

    def test_base_url_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "  http://host:3000  ")
        assert onebot_base_url() == "http://host:3000"

    def test_base_url_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.delenv("ONEBOT_WS_URL", raising=False)
        assert onebot_base_url() == ""

    def test_base_url_falls_back_to_ws_url(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.setenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
        assert onebot_base_url() == "ws://127.0.0.1:3001"

    def test_http_url_wins_over_ws_url(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        monkeypatch.setenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
        assert onebot_base_url() == "http://127.0.0.1:3000"

    def test_access_token_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "  tok123  ")
        assert onebot_access_token() == "tok123"

    def test_access_token_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        assert onebot_access_token() == ""


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------

class TestConfigured:
    def test_configured_when_http_url_set(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        assert onebot_configured() is True

    def test_configured_when_only_ws_url_set(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.setenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
        assert onebot_configured() is True

    def test_not_configured_when_unset(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.delenv("ONEBOT_WS_URL", raising=False)
        assert onebot_configured() is False

    def test_not_configured_when_blank(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "   ")
        monkeypatch.delenv("ONEBOT_WS_URL", raising=False)
        assert onebot_configured() is False


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

class TestOnebotCallHTTP:
    def test_raises_when_url_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.delenv("ONEBOT_WS_URL", raising=False)
        with pytest.raises(RuntimeError, match="ONEBOT_HTTP_URL"):
            onebot_call("get_login_info")

    def test_returns_data_on_success(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps(
            {"status": "ok", "retcode": 0, "data": {"message_id": 7}}
        ).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            data = onebot_call("send_msg", {"message": []})
        assert data == {"message_id": 7}

    def test_raises_on_failed_status(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps(
            {"status": "failed", "retcode": 1404, "message": "no login"}
        ).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            with pytest.raises(RuntimeError, match="no login"):
                onebot_call("send_msg")

    def test_raises_on_missing_data(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps({"status": "ok", "retcode": 0}).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            with pytest.raises(RuntimeError, match="no data"):
                onebot_call("send_msg")

    def test_raises_on_non_json(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse(b"<html>502 Bad Gateway</html>"),
        ):
            with pytest.raises(RuntimeError, match="non-JSON"):
                onebot_call("send_msg")

    def test_auth_header_sent_when_token_configured(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "secret")
        body = json.dumps({"status": "ok", "data": {"ok": 1}}).encode()
        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["auth"] = req.headers.get("Authorization")
            return _FakeHTTPResponse(body)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            onebot_call("get_status")
        assert captured["auth"] == "Bearer secret"


# ---------------------------------------------------------------------------
# WebSocket transport
# ---------------------------------------------------------------------------

class TestOnebotCallWS:
    def test_returns_data_on_success(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "ws://127.0.0.1:3001")
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        fake = _FakeWS(data={"user_id": 42}, lead_events=3)
        with patch("websockets.connect", _fake_connect(fake)):
            data = onebot_call("get_login_info")
        assert data == {"user_id": 42}
        # The action request was actually sent.
        sent = json.loads(fake.sent[-1])
        assert sent["action"] == "get_login_info"

    def test_uses_ws_url_fallback(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        monkeypatch.setenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        fake = _FakeWS(data={"message_id": 9})
        with patch("websockets.connect", _fake_connect(fake)):
            data = onebot_call("send_msg", {"message": []})
        assert data == {"message_id": 9}

    def test_raises_on_failed_status(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "ws://127.0.0.1:3001")
        fake = _FakeWS(status="failed", retcode=1400, message="bad request")
        with patch("websockets.connect", _fake_connect(fake)):
            with pytest.raises(RuntimeError, match="bad request"):
                onebot_call("send_msg")

    def test_raises_on_missing_data(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "ws://127.0.0.1:3001")
        fake = _FakeWS(data=None)  # status ok but no data field
        with patch("websockets.connect", _fake_connect(fake)):
            with pytest.raises(RuntimeError, match="no data"):
                onebot_call("get_login_info")

    def test_access_token_in_uri(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "ws://127.0.0.1:3001")
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "s3cr3t")
        fake = _FakeWS(data={"ok": 1})
        connect = _fake_connect(fake)
        with patch("websockets.connect", connect):
            onebot_call("get_status")
        assert "access_token=s3cr3t" in connect.last_uri

    def test_no_access_token_in_uri_when_absent(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "ws://127.0.0.1:3001")
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        fake = _FakeWS(data={"ok": 1})
        connect = _fake_connect(fake)
        with patch("websockets.connect", connect):
            onebot_call("get_status")
        assert "access_token" not in connect.last_uri
