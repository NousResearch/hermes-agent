"""Tests for _send_mattermost, _send_matrix, _send_dingtalk."""

import asyncio
import os
import json
import re

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig

# ``_send_dingtalk`` and ``_send_matrix`` moved into their bundled plugins
# (``plugins/platforms/<x>/adapter.py::_standalone_send``) in #41112. Keep
# thin pre-migration-shaped shims so existing test bodies work unchanged.
from plugins.platforms.dingtalk.adapter import (
    _standalone_send as _dingtalk_standalone_send,
)
from plugins.platforms.matrix.standalone import (
    standalone_send as _matrix_standalone_send,
)


async def _send_dingtalk(extra, chat_id, message):
    """Pre-migration ``(extra, chat_id, message)`` shim around the dingtalk
    plugin's ``_standalone_send(pconfig, chat_id, message)``."""
    pconfig = SimpleNamespace(token=None, extra=extra or {})
    return await _dingtalk_standalone_send(pconfig, chat_id, message)


async def _send_matrix(token, extra, chat_id, message):
    """Send through the Matrix plugin with the supplied credentials and settings."""
    pconfig = PlatformConfig(token=token, extra=extra or {})
    return await _matrix_standalone_send(pconfig, chat_id, message)

# ``_send_mattermost`` moved into the mattermost plugin
# (``plugins/platforms/mattermost/adapter.py::_standalone_send``).  Keep a
# thin ``(token, extra, chat_id, message)``-shaped wrapper so existing test
# bodies continue to work without rewriting every signature.
from plugins.platforms.mattermost.adapter import (
    _standalone_send as _mattermost_standalone_send,
)


async def _send_mattermost(token, extra, chat_id, message):
    """Pre-migration ``(token, extra, chat_id, message)`` shim around the
    plugin's ``_standalone_send(pconfig, chat_id, message)``.
    """
    pconfig = SimpleNamespace(token=token, extra=extra or {})
    return await _mattermost_standalone_send(pconfig, chat_id, message)


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
        state = MagicMock()
        state.__aenter__ = AsyncMock(return_value=_make_aiohttp_resp(
            404, json_data={"errcode": "M_NOT_FOUND"},
        ))
        state.__aexit__ = AsyncMock(return_value=False)
        sent = MagicMock()
        sent.__aenter__ = AsyncMock(return_value=resp)
        sent.__aexit__ = AsyncMock(return_value=False)
        session.request = MagicMock(side_effect=[state, sent])

        with patch("aiohttp.ClientSession", return_value=session_ctx) as client_session, \
             patch.dict(os.environ, {"MATRIX_HOMESERVER": "", "MATRIX_ACCESS_TOKEN": ""}, clear=False):
            extra = {"homeserver": "https://matrix.example.com"}
            result = asyncio.run(_send_matrix("syt_tok", extra, "!room:example.com", "hello matrix"))

        assert result == {
            "success": True,
            "platform": "matrix",
            "chat_id": "!room:example.com",
            "message_id": "$abc123",
        }
        assert [call.args[0] for call in session.request.call_args_list] == ["GET", "PUT"]
        call_kwargs = session.request.call_args
        url = call_kwargs[0][1]
        assert url.startswith("https://matrix.example.com/_matrix/client/v3/rooms/%21room%3Aexample.com/send/m.room.message/")
        assert client_session.call_args.kwargs["headers"] == {"Authorization": "Bearer syt_tok"}
        payload = call_kwargs[1]["json"]
        assert payload["msgtype"] == "m.text"
        assert payload["body"] == "hello matrix"

    def test_standalone_markdown_html_is_sanitized(self):
        resp = _make_aiohttp_resp(200, json_data={"event_id": "$safe"})
        session_ctx, session = _make_aiohttp_session(resp)
        members = _make_aiohttp_resp(200, json_data={
            "joined": {"@bot:example.com": {}, "@alice:example.com": {}, "@bob:example.com": {}},
        })
        responses = iter((
            _make_aiohttp_resp(404, json_data={"errcode": "M_NOT_FOUND"}),
            members, resp, members,
        ))

        def request(method, url, **kwargs):
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=next(responses))
            context.__aexit__ = AsyncMock(return_value=False)
            return context

        session.request = MagicMock(side_effect=request)
        message = (
            "safe <details open onclick=\"alert(1)\"><summary>more</summary>"
            "<script>alert(2)</script></details>"
        )

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_matrix(
                    "tok",
                    {"homeserver": "https://matrix.example.com"},
                    "!room:example.com",
                    message,
                )
            )

        assert result["success"] is True
        payload = session.request.call_args_list[2].kwargs["json"]
        assert payload["msgtype"] == "m.text"
        assert payload["format"] == "org.matrix.custom.html"
        assert "<details>" in payload["formatted_body"]
        assert "<summary>more</summary>" in payload["formatted_body"]
        assert "<script" not in payload["formatted_body"]
        assert "alert(" not in payload["formatted_body"]
        assert " onclick=" not in payload["formatted_body"]

    def test_txn_id_is_unique_across_calls(self):
        """Each call should generate a distinct transaction ID in the URL."""
        txn_ids = []

        def capture(method, url, **kwargs):
            ctx = MagicMock()
            if method == "GET":
                response = _make_aiohttp_resp(404, json_data={"errcode": "M_NOT_FOUND"})
            else:
                txn_ids.append(url.rsplit("/", 1)[-1])
                response = _make_aiohttp_resp(200, json_data={"event_id": "$x"})
            ctx.__aenter__ = AsyncMock(return_value=response)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session = MagicMock()
        session.request = capture
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        extra = {"homeserver": "https://matrix.example.com"}

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            asyncio.run(_send_matrix("tok", extra, "!r:example.com", "first"))
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            asyncio.run(_send_matrix("tok", extra, "!r:example.com", "second"))

        assert len(txn_ids) == 2
        assert txn_ids[0] != txn_ids[1]


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


    def test_env_var_fallback(self):
        resp = self._make_httpx_resp(json_data={"errcode": 0, "errmsg": "ok"})
        client_ctx, client = self._make_httpx_client(resp)

        with patch("httpx.AsyncClient", return_value=client_ctx), \
             patch.dict(os.environ, {"DINGTALK_WEBHOOK_URL": "https://oapi.dingtalk.com/robot/send?access_token=env"}, clear=False):
            result = asyncio.run(_send_dingtalk({}, "ch", "hi"))

        assert result["success"] is True
        call_kwargs = client.post.await_args
        assert "access_token=env" in call_kwargs[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["expanded-html", "unicode-chunks", "thread-boundary", "escaped-controls"])
async def test_standalone_transport_payloads_fit_the_matrix_event_limit(monkeypatch, case):
    from plugins.platforms.matrix.standalone import _HTTPDelivery, _MatrixAPIError

    payloads = []

    async def request(self, method, path, **kwargs):
        if path.endswith("/state/m.room.encryption"):
            raise _MatrixAPIError(404, {"errcode": "M_NOT_FOUND"})
        if path.endswith("/joined_members"):
            return {"joined": {"@bot:server": {}, "@user:server": {}, "@other:server": {}}}
        payloads.append(kwargs["json"])
        return {"event_id": "$sent"}

    monkeypatch.setattr(_HTTPDelivery, "request", request)
    message = {"expanded-html": "safe", "unicode-chunks": "🧪" * 16000,
               "thread-boundary": "&" * 7484, "escaped-controls": "\0" * 8000}[case]
    thread_id = "$" + "t" * 200 if case == "thread-boundary" else "$root"
    if case == "expanded-html":
        monkeypatch.setattr("markdown.markdown", lambda *_args, **_kwargs: "<p>" + "x" * 45000 + "</p>")
    delivery = _HTTPDelivery(None, "https://matrix.test")
    await delivery.send("!room:server", message, thread_id=thread_id)

    assert payloads
    bodies = [re.sub(r" \(\d+/\d+\)$", "", payload["body"]) for payload in payloads]
    assert "".join(bodies) == message
    assert all(len(json.dumps(payload).encode("utf-8")) <= 45000
               for payload in payloads)
    assert [payload["m.relates_to"] for payload in payloads] == [{
        "rel_type": "m.thread", "event_id": thread_id, "is_falling_back": True,
        "m.in_reply_to": {"event_id": thread_id},
    }] * len(payloads)
    if case == "expanded-html":
        assert payloads == [{"msgtype": "m.text", "body": "safe", "m.relates_to": {
            "rel_type": "m.thread", "event_id": thread_id, "is_falling_back": True,
            "m.in_reply_to": {"event_id": thread_id},
        }}]
