"""Tests for send_message plugin enrichers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.send_message_tool import (
    SEND_MESSAGE_SCHEMA,
    _SEND_MESSAGE_ENRICHERS,
    _SEND_MESSAGE_SCHEMA_FRAGMENTS,
    _parse_target_ref,
    _send_to_platform,
    get_send_message_schema,
    register_send_message_enricher,
)


@pytest.fixture(autouse=True)
def _reset_enrichers():
    """Clear the enricher registry before and after every test."""
    _SEND_MESSAGE_ENRICHERS.clear()
    _SEND_MESSAGE_SCHEMA_FRAGMENTS.clear()
    yield
    _SEND_MESSAGE_ENRICHERS.clear()
    _SEND_MESSAGE_SCHEMA_FRAGMENTS.clear()


class TestParseTargetRef:
    def test_enricher_target_parsing(self):
        """Plugin-enriched platforms treat the target_ref as explicit."""
        register_send_message_enricher("myplatform", AsyncMock())
        chat_id, thread_id, is_explicit = _parse_target_ref("myplatform", "+123")
        assert chat_id == "+123"
        assert thread_id is None
        assert is_explicit is True

    def test_builtin_precedence(self):
        """Built-in platforms win over enrichers with the same name."""
        register_send_message_enricher("telegram", AsyncMock())
        chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "777")
        assert chat_id == "777"
        assert thread_id is None
        assert is_explicit is True

    def test_enricher_empty_target_ref(self):
        """Empty target_ref for an enriched platform is not explicit."""
        register_send_message_enricher("myplatform", AsyncMock())
        chat_id, thread_id, is_explicit = _parse_target_ref("myplatform", "")
        assert chat_id is None
        assert thread_id is None
        assert is_explicit is False


class TestPlatformEntryTargetParsing:
    def test_parse_target_ref_fn_is_used(self):
        """PlatformEntry.parse_target_ref_fn is consulted for explicit targets."""
        from gateway.platform_registry import PlatformEntry, platform_registry

        platform_name = "fmsg-parse-test"

        def _parser(ref):
            if ref.startswith("@") and "@" in ref[1:]:
                return ref, None
            return None

        entry = PlatformEntry(
            name=platform_name,
            label="Fmsg parse test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            parse_target_ref_fn=_parser,
        )
        platform_registry.register(entry)
        try:
            chat_id, thread_id, is_explicit = _parse_target_ref(
                platform_name, "@alice@example.com"
            )
            assert chat_id == "@alice@example.com"
            assert thread_id is None
            assert is_explicit is True
        finally:
            platform_registry.unregister(platform_name)

    def test_parse_target_ref_fn_returns_none_falls_through(self):
        """When parse_target_ref_fn returns None, target falls through to enricher."""
        from gateway.platform_registry import PlatformEntry, platform_registry

        platform_name = "fmsg-fallback-test"

        def _parser(ref):
            return None

        entry = PlatformEntry(
            name=platform_name,
            label="Fmsg fallback test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            parse_target_ref_fn=_parser,
        )
        platform_registry.register(entry)
        register_send_message_enricher(platform_name, AsyncMock())
        try:
            chat_id, thread_id, is_explicit = _parse_target_ref(
                platform_name, "anything"
            )
            assert chat_id == "anything"
            assert thread_id is None
            assert is_explicit is True
        finally:
            platform_registry.unregister(platform_name)

    def test_parse_target_ref_fn_thread_id(self):
        """parse_target_ref_fn can return a thread_id alongside chat_id."""
        from gateway.platform_registry import PlatformEntry, platform_registry

        platform_name = "threaded-parse-test"

        def _parser(ref):
            if ":" in ref:
                room, thread = ref.split(":", 1)
                return room, thread
            return None

        entry = PlatformEntry(
            name=platform_name,
            label="Threaded parse test",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            parse_target_ref_fn=_parser,
        )
        platform_registry.register(entry)
        try:
            chat_id, thread_id, is_explicit = _parse_target_ref(
                platform_name, "room-1:thread-99"
            )
            assert chat_id == "room-1"
            assert thread_id == "thread-99"
            assert is_explicit is True
        finally:
            platform_registry.unregister(platform_name)

    def test_builtin_platform_ignores_parse_target_ref_fn(self):
        """Built-in platform parsers win over a rogue PlatformEntry parser."""
        from gateway.platform_registry import PlatformEntry, platform_registry

        def _parser(ref):
            return "SHOULD_NOT_RETURN", None

        entry = PlatformEntry(
            name="telegram",
            label="Telegram rogue",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            parse_target_ref_fn=_parser,
        )
        platform_registry.register(entry)
        try:
            chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "12345")
            assert chat_id == "12345"
            assert thread_id is None
            assert is_explicit is True
        finally:
            platform_registry.unregister("telegram")


class TestSchemaMerge:
    def test_schema_fragment_stored(self):
        """Schema fragments are stored in _SEND_MESSAGE_SCHEMA_FRAGMENTS."""
        register_send_message_enricher(
            "myplatform",
            AsyncMock(),
            schema_fragment={"voice": {"type": "string", "description": "Voice setting"}},
        )
        assert "myplatform" in _SEND_MESSAGE_SCHEMA_FRAGMENTS
        assert _SEND_MESSAGE_SCHEMA_FRAGMENTS["myplatform"]["voice"]["type"] == "string"

    def test_get_send_message_schema_assembles_fragments(self):
        """get_send_message_schema returns a copy with fragments merged."""
        register_send_message_enricher(
            "myplatform",
            AsyncMock(),
            schema_fragment={"voice": {"type": "string", "description": "Voice setting"}},
        )
        schema = get_send_message_schema()
        assert "voice" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["voice"]["type"] == "string"

    def test_original_schema_not_mutated(self):
        """The module-level SEND_MESSAGE_SCHEMA is never mutated."""
        register_send_message_enricher(
            "myplatform",
            AsyncMock(),
            schema_fragment={"voice": {"type": "string", "description": "Voice setting"}},
        )
        assert "voice" not in SEND_MESSAGE_SCHEMA["parameters"]["properties"]

    def test_schema_fragment_without_registration(self):
        """No fragment is added when schema_fragment is omitted."""
        register_send_message_enricher("myplatform", AsyncMock())
        assert "myplatform" not in _SEND_MESSAGE_SCHEMA_FRAGMENTS


class TestHandlerRouting:
    def test_async_handler_invoked(self):
        """The async enricher handler receives correct arguments."""
        handler = AsyncMock(return_value={"success": True, "message_id": "msg_1"})
        register_send_message_enricher("myplatform", handler)

        pconfig = SimpleNamespace(enabled=True, token="tok", extra={})
        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                pconfig,
                "chat42",
                "hello",
                args={"target": "myplatform:chat42", "message": "hello", "intent": "greet"},
            )
        )

        assert result == {"success": True, "message_id": "msg_1"}
        handler.assert_awaited_once()
        call_args = handler.await_args.args
        assert call_args[0] == {"target": "myplatform:chat42", "message": "hello", "intent": "greet"}
        assert call_args[1] == "chat42"
        assert call_args[2] == "myplatform"
        assert call_args[3] is pconfig

    def test_sync_handler_invoked(self):
        """The sync enricher handler is called directly (not awaited)."""
        handler = MagicMock(return_value={"success": True, "message_id": "msg_sync"})
        register_send_message_enricher("myplatform", handler)

        pconfig = SimpleNamespace(enabled=True, token="tok", extra={})
        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                pconfig,
                "chat42",
                "hello",
                args={"target": "myplatform:chat42", "message": "hello"},
            )
        )

        assert result == {"success": True, "message_id": "msg_sync"}
        handler.assert_called_once()
        call_args = handler.call_args.args
        assert call_args[0] == {"target": "myplatform:chat42", "message": "hello"}
        assert call_args[1] == "chat42"
        assert call_args[2] == "myplatform"
        assert call_args[3] is pconfig

    def test_async_handler_result(self):
        """Async enricher result is returned verbatim."""
        handler = AsyncMock(return_value={"success": True, "message_id": "msg_1"})
        register_send_message_enricher("myplatform", handler)

        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                SimpleNamespace(enabled=True, token="tok", extra={}),
                "chat42",
                "hello",
            )
        )
        assert result == {"success": True, "message_id": "msg_1"}

    def test_sync_handler_result(self):
        """Sync enricher result is returned verbatim."""
        handler = MagicMock(return_value={"success": True, "message_id": "msg_sync"})
        register_send_message_enricher("myplatform", handler)

        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                SimpleNamespace(enabled=True, token="tok", extra={}),
                "chat42",
                "hello",
            )
        )
        assert result == {"success": True, "message_id": "msg_sync"}

    def test_async_handler_error(self):
        """Async enricher exceptions are caught and surfaced as error dicts."""
        handler = AsyncMock(side_effect=RuntimeError("boom"))
        register_send_message_enricher("myplatform", handler)

        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                SimpleNamespace(enabled=True, token="tok", extra={}),
                "chat42",
                "hello",
            )
        )
        assert "error" in result
        assert "boom" in result["error"]

    def test_sync_handler_error(self):
        """Sync enricher exceptions are caught and surfaced as error dicts."""
        handler = MagicMock(side_effect=RuntimeError("sync boom"))
        register_send_message_enricher("myplatform", handler)

        result = asyncio.run(
            _send_to_platform(
                "myplatform",
                SimpleNamespace(enabled=True, token="tok", extra={}),
                "chat42",
                "hello",
            )
        )
        assert "error" in result
        assert "sync boom" in result["error"]


class TestFallback:
    def test_no_enricher_falls_through(self):
        """Platforms without an enricher still reach _send_via_adapter."""
        result = asyncio.run(
            _send_to_platform(
                "unknownplatform",
                SimpleNamespace(enabled=True, token="tok", extra={}),
                "chat42",
                "hello",
            )
        )
        # _send_via_adapter will error because no adapter is registered;
        # the exact error text is not important, just that it didn't crash
        # inside an enricher branch.
        assert isinstance(result, dict)
        assert "error" in result
