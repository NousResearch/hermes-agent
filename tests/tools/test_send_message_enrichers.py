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
