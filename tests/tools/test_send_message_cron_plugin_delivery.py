"""Regression tests for #87634 — cron delivery payload preservation for plugin platforms.

Ensures:
1. When a plugin registers both send_message_handler and standalone_sender_fn,
   cron delivery (no args) routes to standalone_sender_fn with complete message,
   thread, and media data.
2. An interactive send_message call (with args) continues to route to send_message_handler.
3. When a plugin registers only send_message_handler, cron delivery passes synthesized
   request data so content is never dropped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import _send_to_platform


@pytest.fixture
def _custom_plugin_platform():
    platform_name = "test_custom_plugin"
    custom_platform = Platform(platform_name) if platform_name in Platform._value2member_map_ else platform_name
    yield custom_platform, platform_name
    platform_registry._entries.pop(platform_name, None)


def test_cron_delivery_routes_to_standalone_sender_when_both_registered(_custom_plugin_platform):
    """Cron delivery (args=None) routes to standalone_sender_fn and preserves all payload fields."""
    custom_platform, platform_name = _custom_plugin_platform

    handler_calls = []
    standalone_calls = []

    async def _mock_handler(args, chat_id, p_name, pconfig):
        handler_calls.append((args, chat_id, p_name))
        return {"success": True, "handler": True}

    async def _mock_standalone(pconfig, chat_id, chunk, thread_id=None, media_files=None, force_document=False):
        standalone_calls.append({
            "chat_id": chat_id,
            "chunk": chunk,
            "thread_id": thread_id,
            "media_files": media_files,
            "force_document": force_document,
        })
        return {"success": True, "standalone": True}
    entry = PlatformEntry(
        name=platform_name,
        label="Test Plugin",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        send_message_handler=_mock_handler,
        standalone_sender_fn=_mock_standalone,
    )
    platform_registry.register(entry)

    pconfig = SimpleNamespace(enabled=True, extra={})

    # Cron delivery shape (no args)
    result = asyncio.run(
        _send_to_platform(
            custom_platform,
            pconfig,
            "channel-42",
            "cron report body",
            thread_id="topic-99",
            media_files=[("doc.pdf", False)],
            force_document=True,
        )
    )

    assert result.get("success") is True
    assert len(handler_calls) == 0, "send_message_handler should not intercept cron delivery"
    assert len(standalone_calls) == 1
    call = standalone_calls[0]
    assert call["chat_id"] == "channel-42"
    assert call["chunk"] == "cron report body"
    assert call["thread_id"] == "topic-99"
    assert call["media_files"] == [("doc.pdf", False)]
    assert call["force_document"] is True


def test_interactive_delivery_routes_to_send_message_handler_with_args(_custom_plugin_platform):
    """Interactive send_message (with args dict) routes to send_message_handler."""
    custom_platform, platform_name = _custom_plugin_platform

    handler_calls = []
    standalone_calls = []

    async def _mock_handler(args, chat_id, p_name, pconfig):
        handler_calls.append((args, chat_id, p_name))
        return {"success": True, "handler": True}

    async def _mock_standalone(pconfig, chat_id, chunk, thread_id=None, media_files=None, force_document=False):
        standalone_calls.append(chunk)
        return {"success": True}
    entry = PlatformEntry(
        name=platform_name,
        label="Test Plugin",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        send_message_handler=_mock_handler,
        standalone_sender_fn=_mock_standalone,
    )
    platform_registry.register(entry)

    pconfig = SimpleNamespace(enabled=True, extra={})
    raw_args = {"target": "channel-42", "message": "hello", "custom_opt": 123}

    result = asyncio.run(
        _send_to_platform(
            custom_platform,
            pconfig,
            "channel-42",
            "hello",
            args=raw_args,
        )
    )

    assert result.get("success") is True
    assert len(handler_calls) == 1
    assert handler_calls[0][0] == raw_args
    assert len(standalone_calls) == 0


def test_cron_delivery_synthesizes_args_when_only_handler_registered(_custom_plugin_platform):
    """When a plugin registers only send_message_handler, cron delivery synthesizes payload args."""
    custom_plugin, platform_name = _custom_plugin_platform

    handler_calls = []

    async def _mock_handler(args, chat_id, p_name, pconfig):
        handler_calls.append(args)
        return {"success": True, "synthesized": True}
    entry = PlatformEntry(
        name=platform_name,
        label="Test Plugin",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        send_message_handler=_mock_handler,
        standalone_sender_fn=None,
    )
    platform_registry.register(entry)

    pconfig = SimpleNamespace(enabled=True, extra={})

    result = asyncio.run(
        _send_to_platform(
            custom_plugin,
            pconfig,
            "chan-7",
            "cron payload",
            thread_id="th-1",
            media_files=[("img.png", False)],
        )
    )

    assert result.get("success") is True
    assert len(handler_calls) == 1
    synth = handler_calls[0]
    assert synth["message"] == "cron payload"
    assert synth["chat_id"] == "chan-7"
    assert synth["thread_id"] == "th-1"
    assert synth["media_files"] == [("img.png", False)]
