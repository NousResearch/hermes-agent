"""QQ Bot standalone text sends honour markdown_support like the gateway adapter (#137213)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from tools.send_message_tool import _send_qqbot


def _pconfig(extra=None):
    merged = {"app_id": "app-123", "client_secret": "client-secret"}
    merged.update(extra or {})
    return SimpleNamespace(token="client-secret", extra=merged, enabled=True)


def _token_client(post_handler):
    """httpx.AsyncClient fake whose ``post`` delegates to ``post_handler(url, json)``."""

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            return post_handler(url, json)

    class _HttpxMod:
        AsyncClient = _Client

    return _HttpxMod()


class TestQqbotStandaloneMarkdown:
    def test_text_defaults_to_markdown_msg_type(self):
        seen = []

        def _post(url, json):
            if "getAppAccessToken" in url:
                return SimpleNamespace(
                    status_code=200, json=lambda: {"access_token": "at-1"}
                )
            seen.append((url, json))
            return SimpleNamespace(status_code=200, json=lambda: {"id": "m-1"})

        with patch.dict("sys.modules", {"httpx": _token_client(_post)}):
            result = asyncio.run(_send_qqbot(_pconfig(), "user-openid", "# Title"))

        assert result["success"] is True
        url, payload = seen[0]
        assert "channels/user-openid/messages" in url
        assert payload["msg_type"] == 2
        assert payload["markdown"]["content"] == "# Title"

    def test_markdown_support_false_sends_plain_text(self):
        seen = []

        def _post(url, json):
            if "getAppAccessToken" in url:
                return SimpleNamespace(
                    status_code=200, json=lambda: {"access_token": "at-1"}
                )
            seen.append((url, json))
            return SimpleNamespace(status_code=200, json=lambda: {"id": "m-2"})

        with patch.dict("sys.modules", {"httpx": _token_client(_post)}):
            result = asyncio.run(
                _send_qqbot(
                    _pconfig({"markdown_support": False}), "user-openid", "# Title"
                )
            )

        assert result["success"] is True
        _, payload = seen[0]
        assert payload["msg_type"] == 0
        assert payload["content"] == "# Title"
        assert "markdown" not in payload

    def test_markdown_rejected_by_channel_falls_back_to_c2c(self):
        seen = []

        def _post(url, json):
            if "getAppAccessToken" in url:
                return SimpleNamespace(
                    status_code=200, json=lambda: {"access_token": "at-1"}
                )
            seen.append((url, json))
            if "/channels/" in url:
                return SimpleNamespace(
                    status_code=403, json=lambda: {"message": "no template"}
                )
            return SimpleNamespace(status_code=200, json=lambda: {"id": "m-3"})

        with patch.dict("sys.modules", {"httpx": _token_client(_post)}):
            result = asyncio.run(_send_qqbot(_pconfig(), "user-openid", "**bold**"))

        assert result["success"] is True
        assert result["message_id"] == "m-3"
        # The endpoint order is preserved: channel was tried first, c2c won.
        assert len(seen) == 2
        assert "/channels/" in seen[0][0]
        assert "/v2/users/" in seen[1][0]
        # The c2c retry also asked for markdown, not plain text.
        assert seen[1][1]["msg_type"] == 2
