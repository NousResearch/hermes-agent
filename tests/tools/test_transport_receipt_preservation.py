"""Receipt preservation and redaction at tool-facing seams."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult, TransportReceipt, TransportTarget


@pytest.mark.asyncio
async def test_send_via_adapter_preserves_partial_receipts_on_failure():
    from tools.send_message_tool import _send_via_adapter

    target = TransportTarget("matrix", "!room:example.org")
    receipt = TransportReceipt(
        outcome="delivered", provider_message_id="$event-1",
        requested_target=target, actual_target=target,
        component="text", ordinal=0,
    )
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(
        success=False, error="second chunk rejected", receipts=(receipt,),
    )))
    runner = SimpleNamespace(adapters={Platform.MATRIX: adapter})

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = await _send_via_adapter(
            Platform.MATRIX, SimpleNamespace(extra={}),
            "!room:example.org", "message",
        )

    assert result["receipts"] == (receipt,)
    assert "second chunk rejected" in result["error"]


@pytest.mark.asyncio
async def test_standalone_telegram_preserves_every_text_ack_and_target():
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    messages = [SimpleNamespace(message_id=101), SimpleNamespace(message_id=102)]
    with (
        patch("telegram.Bot", return_value=object()),
        patch("plugins.platforms.telegram.adapter.TelegramAdapter.format_message", return_value="formatted"),
        patch("gateway.platforms.base.BasePlatformAdapter.truncate_message", return_value=["one", "two"]),
        patch("tools.send_message_tool._send_telegram_message_with_retry", new=AsyncMock(side_effect=messages)),
    ):
        result = await _send_telegram("test-token", "-100123", "report", thread_id="7")

    assert result["success"] is True
    assert [receipt.provider_message_id for receipt in result["receipts"]] == ["101", "102"]
    assert [receipt.ordinal for receipt in result["receipts"]] == [0, 1]
    assert all(receipt.requested_target.thread_id == "7" for receipt in result["receipts"])
    assert all(receipt.actual_target.thread_id == "7" for receipt in result["receipts"])


@pytest.mark.asyncio
async def test_standalone_telegram_text_ack_uses_inert_provider_id_normalization():
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    string_calls = []

    class HostileProviderId(str):
        def __str__(self):
            string_calls.append("called")
            return "provider-id"

    sender = AsyncMock(
        return_value=SimpleNamespace(message_id=HostileProviderId("provider-id"))
    )
    with (
        patch("telegram.Bot", return_value=object()),
        patch(
            "plugins.platforms.telegram.adapter.TelegramAdapter.format_message",
            return_value="formatted",
        ),
        patch(
            "gateway.platforms.base.BasePlatformAdapter.truncate_message",
            return_value=["one"],
        ),
        patch(
            "tools.send_message_tool._send_telegram_message_with_retry",
            new=sender,
        ),
    ):
        result = await _send_telegram(
            "test-token", "-100123", "report", receipt_bound=True,
        )

    assert result["success"] is True
    assert result["message_id"] == "provider-id"
    assert len(result["receipts"]) == 1
    assert result["receipts"][0].outcome == "delivered"
    assert result["receipts"][0].provider_message_id == "provider-id"
    assert sender.await_count == 1
    assert string_calls == []


@pytest.mark.asyncio
async def test_standalone_telegram_invalid_text_ack_is_unknown_without_magic():
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    magic_calls = []

    class HostileProviderId:
        @property
        def __class__(self):
            magic_calls.append("class")
            return str

        def __str__(self):
            magic_calls.append("str")
            return "spoofed-provider-id"

    sender = AsyncMock(
        return_value=SimpleNamespace(message_id=HostileProviderId())
    )
    with (
        patch("telegram.Bot", return_value=object()),
        patch(
            "plugins.platforms.telegram.adapter.TelegramAdapter.format_message",
            return_value="formatted",
        ),
        patch(
            "gateway.platforms.base.BasePlatformAdapter.truncate_message",
            return_value=["one"],
        ),
        patch(
            "tools.send_message_tool._send_telegram_message_with_retry",
            new=sender,
        ),
    ):
        result = await _send_telegram(
            "test-token", "-100123", "report", receipt_bound=True,
        )

    assert result["error_kind"] == "unknown"
    assert result["retryable"] is False
    assert len(result["receipts"]) == 1
    assert result["receipts"][0].outcome == "unknown"
    assert result["receipts"][0].provider_message_id is None
    assert sender.await_count == 1
    assert magic_calls == []


@pytest.mark.asyncio
async def test_receipt_bound_standalone_telegram_does_not_plaintext_retry_parse_error():
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    sender = AsyncMock(side_effect=RuntimeError("parse entities rejected"))
    with (
        patch("telegram.Bot", return_value=object()),
        patch(
            "plugins.platforms.telegram.adapter.TelegramAdapter.format_message",
            return_value="formatted",
        ),
        patch(
            "gateway.platforms.base.BasePlatformAdapter.truncate_message",
            return_value=["formatted"],
        ),
        patch(
            "tools.send_message_tool._send_telegram_message_with_retry",
            new=sender,
        ),
    ):
        result = await _send_telegram(
            "test-token", "-100123", "report", receipt_bound=True,
        )

    assert "error" in result
    assert sender.await_count == 1


@pytest.mark.asyncio
async def test_standalone_telegram_thread_fallback_keeps_requested_target_truthful():
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    with (
        patch("telegram.Bot", return_value=object()),
        patch("plugins.platforms.telegram.adapter.TelegramAdapter.format_message", return_value="formatted"),
        patch("gateway.platforms.base.BasePlatformAdapter.truncate_message", return_value=["one"]),
        patch(
            "tools.send_message_tool._send_telegram_message_with_retry",
            new=AsyncMock(side_effect=[
                RuntimeError("Message thread not found"),
                SimpleNamespace(message_id=103),
            ]),
        ),
    ):
        result = await _send_telegram("test-token", "-100123", "report", thread_id="7")

    receipt = result["receipts"][0]
    assert receipt.requested_target.thread_id == "7"
    assert receipt.actual_target.thread_id is None


@pytest.mark.asyncio
async def test_standalone_telegram_preserves_media_provider_ack(tmp_path):
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    image = tmp_path / "image.jpg"
    image.write_bytes(b"bounded-image")
    bot = SimpleNamespace(
        send_photo=AsyncMock(return_value=SimpleNamespace(message_id=201)),
    )
    with patch("telegram.Bot", return_value=bot):
        result = await _send_telegram(
            "test-token", "-100123", "", thread_id="7",
            media_files=[(str(image), False)],
        )

    receipt = result["receipts"][0]
    assert receipt.provider_message_id == "201"
    assert receipt.component == "media"
    assert receipt.ordinal == 0
    assert receipt.requested_target.thread_id == "7"
    assert receipt.actual_target.thread_id == "7"


@pytest.mark.asyncio
async def test_standalone_telegram_media_ack_uses_inert_provider_id_normalization(
    tmp_path,
):
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    string_calls = []

    class HostileProviderId(str):
        def __str__(self):
            string_calls.append("called")
            return "media-provider-id"

    image = tmp_path / "image.jpg"
    image.write_bytes(b"bounded-image")
    sender = AsyncMock(
        return_value=SimpleNamespace(
            message_id=HostileProviderId("media-provider-id")
        )
    )
    bot = SimpleNamespace(send_photo=sender)
    with patch("telegram.Bot", return_value=bot):
        result = await _send_telegram(
            "test-token", "-100123", "", thread_id="7",
            media_files=[(str(image), False)], receipt_bound=True,
        )

    assert result["success"] is True
    assert result["message_id"] == "media-provider-id"
    assert len(result["receipts"]) == 1
    assert result["receipts"][0].outcome == "delivered"
    assert result["receipts"][0].provider_message_id == "media-provider-id"
    assert sender.await_count == 1
    assert string_calls == []


@pytest.mark.asyncio
async def test_standalone_telegram_missing_media_caption_fallback_normalizes_ack(
    tmp_path,
):
    pytest.importorskip("telegram")
    from tools.send_message_tool import _send_telegram
    from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: F401

    sender = AsyncMock(return_value=SimpleNamespace(message_id=301))
    bot = SimpleNamespace(send_message=sender)
    with patch("telegram.Bot", return_value=bot):
        result = await _send_telegram(
            "test-token", "-100123", "bounded caption",
            media_files=[(str(tmp_path / "missing.jpg"), False)],
        )

    assert result["success"] is True
    assert result["message_id"] == "301"
    assert sender.await_count == 1


@pytest.mark.asyncio
async def test_matrix_send_core_preserves_typed_receipts_on_success_and_failure():
    from tools.send_message_tool import _matrix_send_core

    target = TransportTarget("matrix", "!room:example.org")
    receipts = tuple(
        TransportReceipt(
            outcome="delivered", provider_message_id=f"$event-{ordinal}",
            requested_target=target, actual_target=target,
            component="text", ordinal=ordinal,
        )
        for ordinal in range(2)
    )
    adapter = SimpleNamespace(send=AsyncMock(return_value=SendResult(
        success=True, message_id="$event-1", receipts=receipts,
    )))
    success = await _matrix_send_core(adapter, target.chat_id, "report", [], None)
    assert success["receipts"] == receipts

    adapter.send = AsyncMock(return_value=SendResult(
        success=False, error="second chunk failed", receipts=(receipts[0],),
    ))
    failed = await _matrix_send_core(adapter, target.chat_id, "report", [], None)
    assert failed["receipts"] == (receipts[0],)


def test_cron_tool_execution_surface_is_bounded_and_redacted(monkeypatch, tmp_path):
    import cron.executions as executions
    from tools.cronjob_tools import _format_job

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    execution = executions.create_execution("tool-job", source="builtin")
    executions.finish_execution(
        execution["id"], success=False,
        error="RAW_TOOL_ERROR_SENTINEL user@example.org /private/report.pdf",
    )

    formatted = _format_job({
        "id": "tool-job", "name": "tool job", "prompt": "normal prompt",
        "schedule_display": "every 1h", "enabled": True,
        "last_status": "error",
        "last_error": "RAW_LAST_ERROR_SENTINEL user@example.org",
        "last_delivery_error": "RAW_DELIVERY_SENTINEL /private/report.pdf",
        "last_fire_error": {
            "at": "2026-08-22T20:00:00+00:00",
            "detail": "RAW_FIRE_SENTINEL provider payload",
        },
        "last_dispatch": {
            "scheduled_at": "2026-09-01T09:00:00+00:00",
            "dispatched_at": "2026-09-01T09:31:00+00:00",
            "lateness_seconds": 1860,
            "kind": "catch_up",
            "detail": "RAW_DISPATCH_SENTINEL provider payload",
        },
    })
    serialized = json.dumps(formatted, sort_keys=True)

    assert formatted["last_execution"] == {
        "status": "failed",
        "receipt": {
            "delivered": 0, "failed": 0, "unknown": 0,
            "targets_delivered": 0,
        },
    }
    assert formatted["last_delivery_error"] == "delivery_failed"
    assert formatted["last_fire_error"] == {
        "at": "2026-08-22T20:00:00+00:00",
        "error_kind": "fire_forward_failed",
    }
    assert formatted["last_dispatch"] == {
        "scheduled_at": "2026-09-01T09:00:00+00:00",
        "dispatched_at": "2026-09-01T09:31:00+00:00",
        "lateness_seconds": 1860.0,
        "kind": "catch_up",
    }
    assert "RAW_TOOL_ERROR_SENTINEL" not in serialized
    assert "RAW_LAST_ERROR_SENTINEL" not in serialized
    assert "RAW_DELIVERY_SENTINEL" not in serialized
    assert "RAW_FIRE_SENTINEL" not in serialized
    assert "RAW_DISPATCH_SENTINEL" not in serialized
    assert "user@example.org" not in serialized
    assert "/private/report.pdf" not in serialized


def test_cron_tool_drops_malformed_fire_timestamp():
    from tools.cronjob_tools import _format_job

    sentinel = "RAW_TOOL_FIRE_AT_SENTINEL user@example.org /private/report.pdf"
    formatted = _format_job({
        "id": "malformed-fire-at",
        "name": "malformed fire",
        "enabled": True,
        "last_fire_error": {"at": sentinel, "detail": "raw detail"},
    })

    assert formatted["last_fire_error"] == {
        "at": None,
        "error_kind": "fire_forward_failed",
    }
    assert sentinel not in json.dumps(formatted, sort_keys=True)


def test_cron_tool_public_projection_rejects_subclasses_before_magic_methods():
    from tools.cronjob_tools import _format_job

    class HostileDict(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError("hostile job get was called")

    class HostileText(str):
        def __bool__(self):
            raise AssertionError("hostile text truthiness was evaluated")

        def __len__(self):
            raise AssertionError("hostile text length was evaluated")

    with pytest.raises(TypeError, match="object"):
        _format_job(HostileDict({"id": "hostile"}))

    public = _format_job({
        "id": "bounded",
        "name": HostileText("private"),
        "deliver": HostileText("external:private"),
        "last_fire_error": HostileDict({"at": "private"}),
    })
    assert public["name"] == "bounded"
    assert public["delivery_kind"] == "local"
    assert public["last_fire_error"] is None
