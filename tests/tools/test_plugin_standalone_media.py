"""Plugin media routing follows the registration contract, not a platform-name list."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import _send_to_platform


def _entry(name, sender, *, standalone_media=False):
    return PlatformEntry(
        name=name, label="Media plugin", adapter_factory=lambda cfg: None,
        check_fn=lambda: True, standalone_sender_fn=sender,
        standalone_media=standalone_media, max_message_length=5,
    )


def test_declared_media_routes_to_standalone_for_captioned_and_media_only(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    sender = AsyncMock(return_value={"success": True, "media_delivered": True})
    platform_registry.register(_entry("test_media_plugin", sender, standalone_media=True))
    try:
        platform = Platform("test_media_plugin")
        config = SimpleNamespace(extra={})
        files = [("/tmp/report.md", False)]
        result = asyncio.run(_send_to_platform(platform, config, "room", "hello world", media_files=files,
                                               force_document=True))
        assert result["success"] and "warnings" not in result
        assert [call.kwargs["media_files"] for call in sender.await_args_list] == [None, files]
        assert sender.await_args_list[-1].kwargs["force_document"] is True

        sender.reset_mock()
        result = asyncio.run(_send_to_platform(platform, config, "room", "", media_files=files))
        assert result["success"] and "warnings" not in result
        sender.assert_awaited_once()
        assert sender.await_args.kwargs["media_files"] == files
    finally:
        platform_registry.unregister("test_media_plugin")


def test_undeclared_plugin_keeps_legacy_media_only_refusal(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    sender = AsyncMock(return_value={"success": True})
    platform_registry.register(_entry("test_text_plugin", sender))
    try:
        result = asyncio.run(_send_to_platform(Platform("test_text_plugin"), SimpleNamespace(extra={}),
                                                "room", "", media_files=[("/tmp/report.md", False)]))
        assert "only media attachments" in result["error"]
        sender.assert_not_awaited()
    finally:
        platform_registry.unregister("test_text_plugin")


def test_declared_media_without_sender_fails_closed(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    platform_registry.register(_entry("test_missing_sender", None, standalone_media=True))
    try:
        result = asyncio.run(_send_to_platform(Platform("test_missing_sender"), SimpleNamespace(extra={}),
                                                "room", "", media_files=[("/tmp/report.md", False)]))
        assert "missing standalone_sender_fn" in result["error"]
    finally:
        platform_registry.unregister("test_missing_sender")
