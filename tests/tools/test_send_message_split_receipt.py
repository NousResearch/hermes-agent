"""A split send must account for every part it delivered.

``_send_chunks`` used to return the last chunk's result and nothing else, so a text that
went out as N messages came back with a single ``message_id`` and a failure after the head
had landed came back as a plain error. These tests pin the receipt: ``total_chunks``,
``message_ids`` and ``accounted_chunks`` on success, ``partial_success`` +
``delivered_message_ids`` when only part of the text reached the chat.

Kept out of ``test_send_message_tool.py`` because that module skips wholesale when the
optional Telegram dependencies are absent.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.platforms.base import SendResult
from tools.send_message_tool import _send_chunks, _send_to_platform, _send_via_adapter

# Two chunks' worth of synthetic text for a 4096-char platform limit.
_LONG_TEXT = "A" * 4000 + "\n\n" + "B" * 2000
_PCONFIG = SimpleNamespace(enabled=True, token=None, extra={})


def _ok(message_id, **extra):
    return {"success": True, "message_id": message_id, **extra}


def _sender(*results):
    """``send_one`` that returns ``results`` in order and records what it was asked to send."""
    sent = []

    async def send_one(chunk, is_last):
        sent.append((chunk, is_last))
        return results[len(sent) - 1]

    return send_one, sent


# --------------------------------------------------------------------------- _send_chunks


def test_split_send_names_every_part() -> None:
    send_one, sent = _sender(_ok("id-1"), _ok("id-2"), _ok("id-3"))

    result = asyncio.run(_send_chunks(["one", "two", "three"], send_one))

    assert len(sent) == 3
    assert result["success"] is True
    assert result["total_chunks"] == 3
    assert result["message_ids"] == ["id-1", "id-2", "id-3"]
    assert result["accounted_chunks"] == 3
    # The pre-existing contract does not move: ``message_id`` stays the last chunk's id.
    assert result["message_id"] == "id-3"


def test_unsplit_send_result_is_unchanged() -> None:
    send_one, sent = _sender(_ok("id-1"))

    result = asyncio.run(_send_chunks(["only"], send_one))

    assert len(sent) == 1
    assert result == {"success": True, "message_id": "id-1"}


def test_failed_tail_names_the_head_already_delivered() -> None:
    send_one, sent = _sender(_ok("id-1"), {"error": "transport closed the connection"})

    result = asyncio.run(_send_chunks(["one", "two", "three"], send_one))

    assert len(sent) == 2, "the send must stop at the failed chunk"
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["delivered_chunks"] == 1
    assert result["total_chunks"] == 3
    assert result["delivered_message_ids"] == ["id-1"]
    assert result["error"] == "transport closed the connection"


def test_first_chunk_failure_is_not_a_partial_delivery() -> None:
    """The twin of the case above: nothing landed, so nothing may be claimed as delivered."""
    send_one, sent = _sender({"error": "channel not found"})

    result = asyncio.run(_send_chunks(["one", "two"], send_one))

    assert len(sent) == 1
    assert "partial_success" not in result
    assert "delivered_message_ids" not in result
    assert "delivered_chunks" not in result
    assert result["error"] == "channel not found"


def test_sender_that_split_again_contributes_all_its_ids() -> None:
    """A chunk's own sender may split further; its ids belong in the receipt too, and the
    id repeated as both ``message_id`` and the last continuation is counted once."""
    send_one, _ = _sender(
        _ok("id-2", continuation_message_ids=("id-1", "id-2")),
        _ok("id-3", message_ids=["id-3", "id-4"]),
    )

    result = asyncio.run(_send_chunks(["one", "two"], send_one))

    assert result["message_ids"] == ["id-1", "id-2", "id-3", "id-4"]
    assert result["total_chunks"] == 2
    # Four ids for two parts: the id count is not a part count, ``accounted_chunks`` is.
    assert result["accounted_chunks"] == 2


def test_accounted_chunks_is_the_completeness_test() -> None:
    """The other side of the case above: a sender that reports success with no id leaves a
    part delivered but unnamed. The receipt must say so instead of letting the shorter id
    list pass for a complete one -- and it must not claim a failure, because the part did go
    out."""
    send_one, _ = _sender(_ok("id-1"), _ok(None))

    result = asyncio.run(_send_chunks(["one", "two"], send_one))

    assert result["success"] is True
    assert result["total_chunks"] == 2
    assert result["message_ids"] == ["id-1"]
    assert result["accounted_chunks"] == 1, "one of the two parts came back without an id"
    assert "partial_success" not in result, "both parts were sent; only a receipt is missing"


def test_chunk_without_an_id_still_counts_as_delivered() -> None:
    """Some senders report success with no id. The head is still visible in the chat, so the
    failure stays a partial delivery even though there is no id to name."""
    send_one, _ = _sender(_ok(None), {"error": "rate limited"})

    result = asyncio.run(_send_chunks(["one", "two"], send_one))

    assert result["partial_success"] is True
    assert result["delivered_chunks"] == 1
    assert result["delivered_message_ids"] == []
    assert result["accounted_chunks"] == 0, "the head landed but cannot be named"


# ----------------------------------------------------------------------- _send_via_adapter


def _adapter_returning(result):
    class Adapter:
        async def send(self, **kwargs):
            return result

    return patch("tools.send_message_tool._live_adapter",
                 return_value=(SimpleNamespace(), Adapter()))


def test_adapter_continuations_reach_the_tool_result() -> None:
    sent = SendResult(success=True, message_id="b-3",
                      continuation_message_ids=("b-2", "b-3"),
                      raw_response={"message_ids": ["b-1", "b-2", "b-3"]})

    with _adapter_returning(sent):
        result = asyncio.run(
            _send_via_adapter(Platform("mattermost"), _PCONFIG, "channel-synthetic", "text"))

    assert result == {"success": True, "message_id": "b-3",
                      "message_ids": ["b-1", "b-2", "b-3"]}


def test_adapter_result_without_raw_response_is_read_safely() -> None:
    """Adapters are duck-typed here: a send result that carries no ``raw_response`` at all
    must not raise while the receipt is being assembled."""
    with _adapter_returning(SimpleNamespace(success=True, message_id="b-1")):
        result = asyncio.run(
            _send_via_adapter(Platform("mattermost"), _PCONFIG, "channel-synthetic", "text"))

    assert result == {"success": True, "message_id": "b-1"}


def test_adapter_result_with_one_id_gains_no_list() -> None:
    with _adapter_returning(SendResult(success=True, message_id="b-1")):
        result = asyncio.run(
            _send_via_adapter(Platform("mattermost"), _PCONFIG, "channel-synthetic", "text"))

    assert result == {"success": True, "message_id": "b-1"}


# ------------------------------------------------------------------------ _send_to_platform


def _routed(sender):
    """Route a generic text send through ``sender``, with a deterministic chunk limit."""
    return (patch("tools.send_message_tool._platform_max_length", return_value=4096),
            patch("tools.send_message_tool._standalone_send_policy", return_value=None),
            patch("tools.send_message_tool._via_adapter_route", sender))


def _run_split_send(sender):
    limit, policy, route = _routed(sender)
    with limit, policy, route:
        return asyncio.run(_send_to_platform(Platform("mattermost"), _PCONFIG,
                                             "channel-synthetic", _LONG_TEXT))


def test_split_text_send_reports_both_parts() -> None:
    chunks = []

    async def sender(platform, pconfig, chat_id, chunk, media, thread_id, force_document):
        chunks.append(chunk)
        return _ok(f"post-{len(chunks)}")

    result = _run_split_send(sender)

    assert len(chunks) == 2, "the synthetic text must exceed the limit"
    assert result["total_chunks"] == 2
    assert result["message_ids"] == ["post-1", "post-2"]
    assert result["accounted_chunks"] == 2


def test_split_text_send_that_loses_the_tail_reports_partial_delivery() -> None:
    chunks = []

    async def sender(platform, pconfig, chat_id, chunk, media, thread_id, force_document):
        chunks.append(chunk)
        if len(chunks) == 1:
            return _ok("post-1")
        return {"error": "transport closed the connection"}

    result = _run_split_send(sender)

    assert len(chunks) == 2
    assert not result.get("success")
    assert result["partial_success"] is True
    assert result["delivered_chunks"] == 1
    assert result["total_chunks"] == 2
    assert result["delivered_message_ids"] == ["post-1"]
