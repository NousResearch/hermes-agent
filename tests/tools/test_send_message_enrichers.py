"""Tests for send_message plugin enrichers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools.send_message_tool import (
    SEND_MESSAGE_SCHEMA,
    _SEND_MESSAGE_ENRICHERS,
    _parse_target_ref,
    _send_to_platform,
    register_send_message_enricher,
)


@pytest.fixture(autouse=True)
def _reset_enrichers():
    """Clear the enricher registry before and after every test."""
    _SEND_MESSAGE_ENRICHERS.clear()
    # Also clear any schema fragments injected by previous tests
    for key in list(SEND_MESSAGE_SCHEMA["parameters"]["properties"]):
        if key not in {
            "action",
            "target",
            "message",
            "emoji",
            "message_id",
        }:
            SEND_MESSAGE_SCHEMA["parameters"]["properties"].pop(key, None)
    yield
    _SEND_MESSAGE_ENRICHERS.clear()
    for key in list(SEND_MESSAGE_SCHEMA["parameters"]["properties"]):
        if key not in {
            "action",
            "target",
            "message",
            "emoji",
            "message_id",
        }:
            SEND_MESSAGE_SCHEMA["parameters"]["properties"].pop(key, None)


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
    def test_schema_fragment_merged(self):
        """Schema fragments are injected into SEND_MESSAGE_SCHEMA."""
        register_send_message_enricher(
            "myplatform",
            AsyncMock(),
            schema_fragment={"voice": {"type": "string", "description": "Voice setting"}},
        )
        props = SEND_MESSAGE_SCHEMA["parameters"]["properties"]
        assert "voice" in props
        assert props["voice"]["type"] == "string"

    def test_schema_fragment_without_registration(self):
        """No fragment is added when schema_fragment is omitted."""
        register_send_message_enricher("myplatform", AsyncMock())
        props = SEND_MESSAGE_SCHEMA["parameters"]["properties"]
        assert "voice" not in props


class TestHandlerRouting:
    def test_enricher_handler_invoked(self):
        """The enricher handler receives correct arguments."""
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

    def test_enricher_handler_result(self):
        """Enricher result is returned verbatim."""
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

    def test_enricher_handler_error(self):
        """Enricher exceptions are caught and surfaced as error dicts."""
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
