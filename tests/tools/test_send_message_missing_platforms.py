"""Tests for _send_mattermost, _send_matrix, _send_homeassistant, _send_dingtalk."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tools.send_message_tool import (
    _send_dingtalk,
    _send_homeassistant,
    _send_mattermost,
    _send_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aiohttp_resp(status, json_data=None, text_data=None):
    """Build a minimal async-context-manager mock for an aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data or "")
    return resp


def _make_aiohttp_session(resp):
    """Wrap a response mock in a session mock that supports async-with for post/put."""
    request_ctx = MagicMock()
    request_ctx.__aenter__ = AsyncMock(return_value=resp)
    request_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=request_ctx)
    session.put = MagicMock(return_value=request_ctx)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, session


# ---------------------------------------------------------------------------
# _send_mattermost
# ---------------------------------------------------------------------------


class TestSendMattermost:
    def test_success(self):
        resp = _make_aiohttp_resp(201, json_data={"id": "post123"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            extra = {"url": "https://mm.example.com"}
            result = asyncio.run(_send_mattermost("tok-abc", extra, "channel1", "hello"))

        assert result == {"success": True, "platform": "mattermost", "chat_id": "channel1", "message_id": "post123"}
        session.post.assert_called_once()
        call_kwargs = session.post.call_args
        assert call_kwargs[0][0] == "https://mm.example.com/api/v4/posts"
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer tok-abc"
        assert call_kwargs[1]["json"] == {"channel_id": "channel1", "message": "hello"}

    def test_http_error(self):
        resp = _make_aiohttp_resp(400, text_data="Bad Request")
        session_ctx, _ = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_mattermost(
                "tok", {"url": "https://mm.example.com"}, "ch", "hi"
            ))

        assert "error" in result
        assert "400" in result["error"]
        assert "Bad Request" in result["error"]

    def test_missing_config(self):
        with patch.dict(os.environ, {"MATTERMOST_URL": "", "MATTERMOST_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_mattermost("", {}, "ch", "hi"))

        assert "error" in result
        assert "MATTERMOST_URL" in result["error"] or "not configured" in result["error"]

    def test_env_var_fallback(self):
        resp = _make_aiohttp_resp(200, json_data={"id": "p99"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATTERMOST_URL": "https://mm.env.com", "MATTERMOST_TOKEN": "env-tok"}, clear=False):
            result = asyncio.run(_send_mattermost("", {}, "ch", "hi"))

        assert result["success"] is True
        call_kwargs = session.post.call_args
        assert "https://mm.env.com" in call_kwargs[0][0]
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer env-tok"


# ---------------------------------------------------------------------------
# _send_matrix
# ---------------------------------------------------------------------------


class TestSendMatrix:
    def test_success(self):
        resp = _make_aiohttp_resp(200, json_data={"event_id": "$abc123"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"MATRIX_HOMESERVER": "", "MATRIX_ACCESS_TOKEN": ""}, clear=False):
            extra = {"homeserver": "https://matrix.example.com"}
            result = asyncio.run(_send_matrix("syt_tok", extra, "!room:example.com", "hello matrix"))

        assert result == {
            "success": True,
            "platform": "matrix",
            "chat_id": "!room:example.com",
            "message_id": "$abc123",
        }
        session.put.assert_called_once()
        call_kwargs = session.put.call_args
        url = call_kwargs[0][0]
        assert url.startswith("https://matrix.example.com/_matrix/client/v3/rooms/!room:example.com/send/m.room.message/")
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer syt_tok"
        assert call_kwargs[1]["json"] == {"msgtype": "m.text", "body": "hello matrix"}

    def test_http_error(self):
        resp = _make_aiohttp_resp(403, text_data="Forbidden")
        session_ctx, _ = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_matrix(
                "tok", {"homeserver": "https://matrix.example.com"},
                "!room:example.com", "hi"
            ))

        assert "error" in result
        assert "403" in result["error"]
        assert "Forbidden" in result["error"]

    def test_missing_config(self):
        with patch.dict(os.environ, {"MATRIX_HOMESERVER": "", "MATRIX_ACCESS_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_matrix("", {}, "!room:example.com", "hi"))

        assert "error" in result
        assert "MATRIX_HOMESERVER" in result["error"] or "not configured" in result["error"]

    def test_env_var_fallback(self):
        resp = _make_aiohttp_resp(200, json_data={"event_id": "$ev1"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {
                 "MATRIX_HOMESERVER": "https://matrix.env.com",
                 "MATRIX_ACCESS_TOKEN": "env-tok",
             }, clear=False):
            result = asyncio.run(_send_matrix("", {}, "!r:env.com", "hi"))

        assert result["success"] is True
        url = session.put.call_args[0][0]
        assert "matrix.env.com" in url

    def test_txn_id_is_unique_across_calls(self):
        """Each call should generate a distinct transaction ID in the URL."""
        txn_ids = []

        def capture(*args, **kwargs):
            url = args[0]
            txn_ids.append(url.rsplit("/", 1)[-1])
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=_make_aiohttp_resp(200, json_data={"event_id": "$x"}))
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session = MagicMock()
        session.put = capture
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        extra = {"homeserver": "https://matrix.example.com"}

        import time
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            asyncio.run(_send_matrix("tok", extra, "!r:example.com", "first"))
        time.sleep(0.002)
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            asyncio.run(_send_matrix("tok", extra, "!r:example.com", "second"))

        assert len(txn_ids) == 2
        assert txn_ids[0] != txn_ids[1]


# ---------------------------------------------------------------------------
# _send_homeassistant
# ---------------------------------------------------------------------------


class TestSendHomeAssistant:
    def test_success(self):
        resp = _make_aiohttp_resp(200)
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"HASS_URL": "", "HASS_TOKEN": ""}, clear=False):
            extra = {"url": "https://hass.example.com"}
            result = asyncio.run(_send_homeassistant("hass-tok", extra, "mobile_app_phone", "alert!"))

        assert result == {"success": True, "platform": "homeassistant", "chat_id": "mobile_app_phone"}
        session.post.assert_called_once()
        call_kwargs = session.post.call_args
        assert call_kwargs[0][0] == "https://hass.example.com/api/services/notify/notify"
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer hass-tok"
        assert call_kwargs[1]["json"] == {"message": "alert!", "target": "mobile_app_phone"}

    def test_http_error(self):
        resp = _make_aiohttp_resp(401, text_data="Unauthorized")
        session_ctx, _ = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_homeassistant(
                "bad-tok", {"url": "https://hass.example.com"},
                "target", "msg"
            ))

        assert "error" in result
        assert "401" in result["error"]
        assert "Unauthorized" in result["error"]

    def test_missing_config(self):
        with patch.dict(os.environ, {"HASS_URL": "", "HASS_TOKEN": ""}, clear=False):
            result = asyncio.run(_send_homeassistant("", {}, "target", "msg"))

        assert "error" in result
        assert "HASS_URL" in result["error"] or "not configured" in result["error"]

    def test_env_var_fallback(self):
        resp = _make_aiohttp_resp(200)
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx), \
             patch.dict(os.environ, {"HASS_URL": "https://hass.env.com", "HASS_TOKEN": "env-tok"}, clear=False):
            result = asyncio.run(_send_homeassistant("", {}, "notify_target", "hi"))

        assert result["success"] is True
        url = session.post.call_args[0][0]
        assert "hass.env.com" in url


# ---------------------------------------------------------------------------
# _send_dingtalk
# ---------------------------------------------------------------------------


class TestSendDingtalk:
    def _make_httpx_resp(self, status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json = MagicMock(return_value=json_data or {"errcode": 0, "errmsg": "ok"})
        resp.raise_for_status = MagicMock()
        return resp

    def _make_httpx_client(self, resp):
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=client)
        client_ctx.__aexit__ = AsyncMock(return_value=False)
        return client_ctx, client

    def test_success(self):
        resp = self._make_httpx_resp(json_data={"errcode": 0, "errmsg": "ok"})
        client_ctx, client = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            extra = {"webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=abc"}
            result = asyncio.run(_send_dingtalk(extra, "ignored", "hello dingtalk"))

        assert result == {"success": True, "platform": "dingtalk", "chat_id": "ignored"}
        client.post.assert_awaited_once()
        call_kwargs = client.post.await_args
        assert call_kwargs[0][0] == "https://oapi.dingtalk.com/robot/send?access_token=abc"
        assert call_kwargs[1]["json"] == {"msgtype": "text", "text": {"content": "hello dingtalk"}}

    def test_api_error_in_response_body(self):
        """DingTalk always returns HTTP 200 but signals errors via errcode."""
        resp = self._make_httpx_resp(json_data={"errcode": 310000, "errmsg": "sign not match"})
        client_ctx, _ = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = asyncio.run(_send_dingtalk(
                {"webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=bad"},
                "ch", "hi"
            ))

        assert "error" in result
        assert "sign not match" in result["error"]

    def test_http_error(self):
        """If raise_for_status throws, the error is caught and returned."""
        resp = self._make_httpx_resp(status_code=429)
        resp.raise_for_status = MagicMock(side_effect=Exception("429 Too Many Requests"))
        client_ctx, _ = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = asyncio.run(_send_dingtalk(
                {"webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=tok"},
                "ch", "hi"
            ))

        assert "error" in result
        assert "DingTalk send failed" in result["error"]

    def test_http_error_redacts_access_token_in_exception_text(self):
        token = "supersecret-access-token-123456789"
        resp = self._make_httpx_resp(status_code=401)
        resp.raise_for_status = MagicMock(
            side_effect=Exception(
                f"POST https://oapi.dingtalk.com/robot/send?access_token={token} returned 401"
            )
        )
        client_ctx, _ = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = asyncio.run(
                _send_dingtalk(
                    {"webhook_url": f"https://oapi.dingtalk.com/robot/send?access_token={token}"},
                    "ch",
                    "hi",
                )
            )

        assert "error" in result
        assert token not in result["error"]
        assert "access_token=***" in result["error"]

    def test_missing_config(self):
        with patch.dict(os.environ, {"DINGTALK_WEBHOOK_URL": ""}, clear=False):
            result = asyncio.run(_send_dingtalk({}, "ch", "hi"))

        assert "error" in result
        assert "DINGTALK_WEBHOOK_URL" in result["error"] or "not configured" in result["error"]

    def test_env_var_fallback(self):
        resp = self._make_httpx_resp(json_data={"errcode": 0, "errmsg": "ok"})
        client_ctx, client = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx), \
             patch.dict(os.environ, {"DINGTALK_WEBHOOK_URL": "https://oapi.dingtalk.com/robot/send?access_token=env"}, clear=False):
            result = asyncio.run(_send_dingtalk({}, "ch", "hi"))

        assert result["success"] is True
        call_kwargs = client.post.await_args
        assert "access_token=env" in call_kwargs[0][0]
