"""Tests for the standalone _send_qqbot payload construction.

Verifies that the three endpoint types (channel, C2C, group) receive the
correct payload format and that msg_seq stays within the 0..65535 range.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# _send_qqbot — standalone QQ Bot HTTP payload
# ---------------------------------------------------------------------------

class TestSendQqbot:
    """Verify _send_qqbot builds correct per-endpoint payloads."""

    MARKDOWN_CONTENT = "**bold** and `code`"

    @staticmethod
    def _mock_token_response():
        """Return a mock that succeeds on the token endpoint."""
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.json.return_value = {"access_token": "test_token", "expires_in": 7200}
        return token_resp

    @staticmethod
    def _make_channel_success():
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "ch_msg_001"}
        return resp

    @staticmethod
    def _make_c2c_success():
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "c2c_msg_001"}
        return resp

    @staticmethod
    def _make_failure():
        resp = MagicMock()
        resp.status_code = 404
        resp.json.return_value = {"code": 1003, "message": "not found"}
        return resp

    # ---------- markdown_enabled=True ----------

    @pytest.mark.asyncio
    async def test_markdown_enabled_channel(self):
        """Channel endpoint gets content-only payload regardless of markdown_support."""
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_qqbot

        pconfig = PlatformConfig(enabled=True, extra={
            "app_id": "app123", "client_secret": "sec456",
            "markdown_support": True,
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # Token succeeds, channel succeeds
            mock_client.post.side_effect = [
                self._mock_token_response(),   # token
                self._make_channel_success(),  # channel → 200
            ]

            result = await _send_qqbot(pconfig, "123456", self.MARKDOWN_CONTENT)

            calls = mock_client.post.call_args_list
            assert len(calls) >= 2

            # First POST is token
            assert "getAppAccessToken" in str(calls[0][0][0])

            # Second POST is channel — must be content-only
            channel_body = calls[1].kwargs["json"]
            assert "content" in channel_body
            assert "msg_seq" not in channel_body
            assert "msg_type" not in channel_body
            assert "markdown" not in channel_body
            assert channel_body["content"] == self.MARKDOWN_CONTENT

            assert result["success"] is True
            assert result["message_id"] == "ch_msg_001"

    @pytest.mark.asyncio
    async def test_markdown_enabled_c2c_fallback(self):
        """When channel fails, C2C endpoint gets markdown payload."""
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_qqbot

        pconfig = PlatformConfig(enabled=True, extra={
            "app_id": "app123", "client_secret": "sec456",
            "markdown_support": True,
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            mock_client.post.side_effect = [
                self._mock_token_response(),
                self._make_failure(),
                self._make_c2c_success(),
            ]

            result = await _send_qqbot(pconfig, "user_001", self.MARKDOWN_CONTENT)

            calls = mock_client.post.call_args_list
            c2c_body = calls[2].kwargs["json"]

            assert "markdown" in c2c_body
            assert c2c_body["markdown"]["content"] == self.MARKDOWN_CONTENT
            assert c2c_body["msg_type"] == 2
            assert 0 <= c2c_body["msg_seq"] <= 65535

            assert result["success"] is True
            assert result["message_id"] == "c2c_msg_001"

    @pytest.mark.asyncio
    async def test_markdown_disabled_channel(self):
        """Channel is always content-only, even with markdown_support=False."""
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_qqbot

        pconfig = PlatformConfig(enabled=True, extra={
            "app_id": "app123", "client_secret": "sec456",
            "markdown_support": False,
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [
                self._mock_token_response(),
                self._make_channel_success(),
            ]

            result = await _send_qqbot(pconfig, "123456", "plain text")

            calls = mock_client.post.call_args_list
            channel_body = calls[1].kwargs["json"]
            assert "content" in channel_body
            assert "msg_type" not in channel_body
            assert "markdown" not in channel_body
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_markdown_disabled_c2c(self):
        """With markdown_support=False, C2C gets text payload + msg_seq."""
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_qqbot

        pconfig = PlatformConfig(enabled=True, extra={
            "app_id": "app123", "client_secret": "sec456",
            "markdown_support": False,
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [
                self._mock_token_response(),
                self._make_failure(),
                self._make_c2c_success(),
            ]

            result = await _send_qqbot(pconfig, "user_001", "plain text")

            calls = mock_client.post.call_args_list
            c2c_body = calls[2].kwargs["json"]

            assert c2c_body["content"] == "plain text"
            assert c2c_body["msg_type"] == 0
            assert 0 <= c2c_body["msg_seq"] <= 65535
            assert "markdown" not in c2c_body

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_msg_seq_range(self):
        """msg_seq is always in 0..65535 for C2C/group."""
        from gateway.config import PlatformConfig
        from tools.send_message_tool import _send_qqbot

        pconfig = PlatformConfig(enabled=True, extra={
            "app_id": "app123", "client_secret": "sec456",
            "markdown_support": True,
        })

        seqs = set()
        for _ in range(20):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client_cls.return_value.__aenter__.return_value = mock_client
                mock_client.post.side_effect = [
                    self._mock_token_response(),
                    self._make_failure(),
                    self._make_c2c_success(),
                ]
                await _send_qqbot(pconfig, "u", "hello")
                calls = mock_client.post.call_args_list
                seq = calls[2].kwargs["json"]["msg_seq"]
                assert 0 <= seq <= 65535, f"msg_seq {seq} out of range"
                seqs.add(seq)

        assert len(seqs) > 1, "msg_seq should vary across calls"
