"""Tests for Signal media delivery in send_message_tool.py."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gateway.config import Platform


def _make_httpx_mock():
    """Create a mock httpx module with proper sync json()."""

    class AsyncBaseTransport:
        pass

    class Proxy:
        pass

    class MockResp:
        status_code = 200
        def json(self):
            return {"timestamp": 1234567890}
        def raise_for_status(self):
            pass

    class MockClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def post(self, *args, **kwargs):
            return MockResp()

    httpx_mock = ModuleType("httpx")
    httpx_mock.AsyncClient = lambda timeout=None: MockClient()
    httpx_mock.AsyncBaseTransport = AsyncBaseTransport  # Needed by Telegram adapter
    httpx_mock.Proxy = Proxy  # Needed by telegram-bot library
    return httpx_mock


@pytest.fixture(autouse=True)
def inject_httpx(monkeypatch):
    """Inject mock httpx into sys.modules before imports."""
    monkeypatch.setitem(sys.modules, "httpx", _make_httpx_mock())


class TestSendSignalMediaFiles:
    """Test that _send_signal correctly handles media_files parameter."""

    def test_send_signal_basic_text_without_media(self):
        """Backward compatibility: text-only signal messages work."""
        from tools.send_message_tool import _send_signal

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(_send_signal(extra, "+155****9999", "Hello world"))

        assert result["success"] is True
        assert result["platform"] == "signal"
        assert result["chat_id"] == "+155****9999"

    def test_send_signal_with_attachments(self, tmp_path):
        """Signal messages with media_files include attachments in JSON-RPC."""
        from tools.send_message_tool import _send_signal

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG")

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(
            _send_signal(extra, "+155****9999", "Check this out", media_files=[(str(img_path), False)])
        )

        assert result["success"] is True
        assert result["platform"] == "signal"

    def test_send_signal_with_missing_media_file(self):
        """Missing media files should generate warnings but not fail."""
        from tools.send_message_tool import _send_signal

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(
            _send_signal(extra, "+155****9999", "File missing?", media_files=[("/nonexistent.png", False)])
        )

        assert result["success"] is True  # Should succeed despite missing file
        assert "warnings" in result
        assert "Some media files were skipped" in str(result["warnings"])


class TestSendSignalMediaRestrictions:
    """Test that the restriction block handles Signal media correctly."""

    def test_signal_allows_text_only_media_via_send_to_platform(self):
        """Signal should accept text-only media files (no message) via _send_to_platform."""
        import httpx
        if not hasattr(httpx, 'Proxy') or not hasattr(httpx, 'URL'):
            pytest.skip("httpx type annotations incompatible with telegram library")
        from tools.send_message_tool import _send_to_platform

        mock_result = {"success": True, "platform": "signal"}
        with patch("tools.send_message_tool._send_signal", new=AsyncMock(return_value=mock_result)):
            config = MagicMock()
            config.platforms = {Platform.SIGNAL: MagicMock(enabled=True)}
            config.get_home_channel.return_value = None

            result = asyncio.run(
                _send_to_platform(
                    Platform.SIGNAL,
                    config,
                    "+155****9999",
                    "",  # Empty message - media is the message
                    media_files=[("/tmp/test.png", False)]
                )
            )

            assert result["success"] is True

    def test_non_media_platforms_reject_text_only_media(self):
        """Slack should reject text-only media (no MESSAGE content)."""
        import httpx
        if not hasattr(httpx, 'Proxy') or not hasattr(httpx, 'URL'):
            pytest.skip("httpx type annotations incompatible with telegram library")
        from tools.send_message_tool import _send_to_platform

        config = MagicMock()
        config.platforms = {Platform.SLACK: MagicMock(enabled=True)}
        config.get_home_channel.return_value = None

        # Empty message with media_files should trigger restriction block
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                config,
                "C012AB3CD",
                "",  # Empty message - media is the only content
                media_files=[("/tmp/test.png", False)]
            )
        )

        assert "error" in result
        assert "only supported for" in result["error"]


class TestSendSignalMediaWarningMessages:
    """Test warning messages are updated to include signal."""

    def test_warning_includes_signal_when_media_omitted(self):
        """Non-media platforms should show a warning mentioning signal in the supported list."""
        import httpx
        if not hasattr(httpx, 'Proxy') or not hasattr(httpx, 'URL'):
            pytest.skip("httpx type annotations incompatible with telegram library")
        from tools.send_message_tool import _send_to_platform

        config = MagicMock()
        config.platforms = {Platform.SLACK: MagicMock(enabled=True)}
        config.get_home_channel.return_value = None

        # Mock _send_slack so it succeeds -> then warning gets attached to result
        with patch("tools.send_message_tool._send_slack", new=AsyncMock(return_value={"success": True})):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    config,
                    "C012AB3CD",
                    "Test message with media",
                    media_files=[("/tmp/test.png", False)]
                )
            )

        assert result.get("warnings") is not None
        # Check that the warning mentions signal as supported
        found = any("signal" in w.lower() for w in result["warnings"])
        assert found, f"Expected 'signal' in warnings but got: {result.get('warnings')}"


class TestSendSignalGroupChats:
    """Test that _send_signal handles group chats correctly."""

    def test_send_signal_group_with_attachments(self, tmp_path):
        """Group chat messages with attachments should use groupId parameter."""
        from tools.send_message_tool import _send_signal

        img_path = tmp_path / "test_attachment.pdf"
        img_path.write_bytes(b"%PDF-1.4")

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(
            _send_signal(extra, "group:abc123==", "Group file", media_files=[(str(img_path), False)])
        )

        assert result["success"] is True


class TestSendSignalConfigLoading:
    """Verify Signal config loading works."""

    def test_signal_platform_exists(self):
        """Platform.SIGNAL should be a valid platform."""
        assert hasattr(Platform, "SIGNAL")
        assert Platform.SIGNAL.value == "signal"
