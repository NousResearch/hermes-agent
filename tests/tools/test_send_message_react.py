"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import tools.send_message_tool as smt


class _FakePhotonAdapter:
    """Adapter exposing add_reaction/remove_reaction coroutines."""

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "emoji": emoji}

    async def remove_reaction(self, chat_id, emoji=None, message_id=None):
        self.calls.append(("remove", chat_id, emoji, message_id))
        return {"success": True}


class _NoReactionAdapter:
    """Adapter with no reaction support at all."""


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("photon"): adapter})


def _call(args):
    return json.loads(smt.send_message_tool(args))


def setup_function():
    smt._reaction_operations.clear()


def test_react_dispatches_to_add_reaction():
    adapter = _FakePhotonAdapter()
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="accept") as consent,
    ):
        result = _call(
            {
                "action": "react",
                "target": "photon:+155****4567",
                "emoji": "❤️",
                "message_id": "message-guid",
            }
        )
    assert result["success"] is True
    assert result["status"] == "applied"
    assert adapter.calls == [("add", "+155****4567", "❤️", "message-guid")]
    shown, description = consent.call_args.args
    assert "❤️" in shown
    assert "photon:+155****4567" in shown
    assert "message-guid" in shown
    assert "approval" in description.lower()


def test_reaction_requires_an_exact_message_and_rejection_sends_nothing():
    adapter = _FakePhotonAdapter()
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="decline"),
    ):
        missing = _call({"action": "react", "target": "photon:+155****4567", "emoji": "👍"})
        rejected = _call(
            {
                "action": "react",
                "target": "photon:+155****4567",
                "emoji": "👍",
                "message_id": "message-guid",
            }
        )

    assert missing["success"] is False
    assert "message_id" in json.dumps(missing)
    assert rejected["success"] is False
    assert rejected["status"] == "rejected"
    assert adapter.calls == []


def test_repeated_approved_operation_is_idempotent():
    adapter = _FakePhotonAdapter()
    args = {
        "action": "react",
        "target": "photon:+155****4567",
        "emoji": "😂",
        "message_id": "message-guid",
    }
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="accept") as consent,
    ):
        first = _call(args)
        replay = _call(args)

    assert first["success"] is True
    assert replay["success"] is True
    assert replay["duplicate"] is True
    assert adapter.calls == [("add", "+155****4567", "😂", "message-guid")]
    consent.assert_called_once()


def test_applied_reaction_can_be_reapplied_after_opposite_state_transition():
    adapter = _FakePhotonAdapter()
    common = {
        "target": "photon:+155****4567",
        "emoji": "😂",
        "message_id": "message-guid",
    }
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch(
            "tools.approval_prompt.request_one_time_consent", return_value="accept"
        ) as consent,
    ):
        added = _call({**common, "action": "react"})
        removed = _call({**common, "action": "unreact"})
        readded = _call({**common, "action": "react"})

    assert added["status"] == removed["status"] == readded["status"] == "applied"
    assert adapter.calls == [
        ("add", "+155****4567", "😂", "message-guid"),
        ("remove", "+155****4567", "😂", "message-guid"),
        ("add", "+155****4567", "😂", "message-guid"),
    ]
    assert consent.call_count == 3


def test_rejected_reaction_can_be_approved_on_a_later_attempt():
    adapter = _FakePhotonAdapter()
    args = {
        "action": "react",
        "target": "photon:+155****4567",
        "emoji": "👍",
        "message_id": "message-guid",
    }
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch(
            "tools.approval_prompt.request_one_time_consent",
            side_effect=["decline", "accept"],
        ) as consent,
    ):
        rejected = _call(args)
        retried = _call(args)

    assert rejected["status"] == "rejected"
    assert retried["status"] == "applied"
    assert adapter.calls == [("add", "+155****4567", "👍", "message-guid")]
    assert consent.call_count == 2


def test_in_flight_reaction_is_not_evicted_by_capacity_pruning():
    adapter = _FakePhotonAdapter()
    first_args = {
        "action": "react",
        "target": "photon:first-chat",
        "emoji": "👍",
        "message_id": "first-message",
    }
    second_args = {
        "action": "react",
        "target": "photon:second-chat",
        "emoji": "❤️",
        "message_id": "second-message",
    }
    nested = None
    prompting_first = True

    def approve_with_nested_operation(_prompt, _description, **_kwargs):
        nonlocal nested, prompting_first
        if prompting_first:
            prompting_first = False
            nested = _call(second_args)
        return "accept"

    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.send_message_tool._REACTION_OPERATION_LIMIT", 1),
        patch(
            "tools.approval_prompt.request_one_time_consent",
            side_effect=approve_with_nested_operation,
        ),
    ):
        first = _call(first_args)

    assert nested is not None and nested["status"] == "applied"
    assert first["status"] == "applied"
    assert adapter.calls == [
        ("add", "second-chat", "❤️", "second-message"),
        ("add", "first-chat", "👍", "first-message"),
    ]


def test_rapid_repeat_while_approval_is_pending_sends_nothing_early():
    adapter = _FakePhotonAdapter()
    args = {
        "action": "react",
        "target": "photon:+155****4567",
        "emoji": "👍",
        "message_id": "message-guid",
    }
    replay = None

    def approve_after_replay(_prompt, _description, **_kwargs):
        nonlocal replay
        assert adapter.calls == []
        replay = _call(args)
        assert adapter.calls == []
        return "accept"

    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", side_effect=approve_after_replay),
    ):
        first = _call(args)

    assert first["status"] == "applied"
    assert replay is not None
    assert replay["status"] == "pending"
    assert replay["duplicate"] is True
    assert adapter.calls == [("add", "+155****4567", "👍", "message-guid")]


def test_exact_chat_is_part_of_outbound_idempotency_identity():
    adapter = _FakePhotonAdapter()
    common = {"action": "react", "emoji": "❤️", "message_id": "reused-guid"}
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="accept") as consent,
    ):
        first = _call({**common, "target": "photon:first-chat"})
        second = _call({**common, "target": "photon:second-chat"})

    assert first["status"] == second["status"] == "applied"
    assert adapter.calls == [
        ("add", "first-chat", "❤️", "reused-guid"),
        ("add", "second-chat", "❤️", "reused-guid"),
    ]
    assert consent.call_count == 2


def test_unreact_keeps_reaction_identity_and_exact_target():
    adapter = _FakePhotonAdapter()
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="accept"),
    ):
        result = _call(
            {
                "action": "unreact",
                "target": "photon:+155****4567",
                "emoji": "‼️",
                "message_id": "message-guid",
            }
        )

    assert result["status"] == "applied"
    assert adapter.calls == [("remove", "+155****4567", "‼️", "message-guid")]


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {
                "action": "react",
                "target": "photon:+155****4567",
                "emoji": "👍",
                "message_id": "message-guid",
            }
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)


def test_unsupported_adapter_fails_before_requesting_approval():
    adapter = _NoReactionAdapter()
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent") as consent,
    ):
        result = _call(
            {
                "action": "react",
                "target": "photon:+155****4567",
                "emoji": "👍",
                "message_id": "message-guid",
            }
        )

    assert result.get("success") is not True
    assert "does not support" in result["error"]
    consent.assert_not_called()


def test_failed_provider_operation_can_be_reapproved_and_retried():
    class FlakyAdapter(_FakePhotonAdapter):
        async def add_reaction(self, chat_id, emoji, message_id=None):
            self.calls.append(("add", chat_id, emoji, message_id))
            return {"success": len(self.calls) > 1, "error": "temporary failure"}

    adapter = FlakyAdapter()
    args = {
        "action": "react",
        "target": "photon:+155****4567",
        "emoji": "👍",
        "message_id": "message-guid",
    }
    with (
        patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)),
        patch("tools.approval_prompt.request_one_time_consent", return_value="accept") as consent,
    ):
        failed = _call(args)
        retried = _call(args)

    assert failed["success"] is False
    assert failed["status"] == "failed"
    assert retried["success"] is True
    assert retried["status"] == "applied"
    assert adapter.calls == [
        ("add", "+155****4567", "👍", "message-guid"),
        ("add", "+155****4567", "👍", "message-guid"),
    ]
    assert consent.call_count == 2
