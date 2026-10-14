"""Tests for Signal media delivery in send_message_tool.py."""

import asyncio
import sys
from types import ModuleType

import pytest


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


    def test_send_signal_with_missing_media_file(self):
        """Missing media files should generate warnings but not fail."""
        from tools.send_message_tool import _send_signal

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(
            _send_signal(extra, "+155****9999", "File missing?", media_files=[("/nonexistent.png", False)])
        )

        assert result["success"] is True  # Should succeed despite missing file
        assert result["warnings"]
