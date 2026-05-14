import json
import types
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session_context import set_session_vars


@pytest.fixture
def matrix_tool_context(monkeypatch):
    for name in (
        "MATRIX_TOOLS_ALLOW_REDACTION",
        "MATRIX_TOOLS_ALLOW_INVITES",
        "MATRIX_TOOLS_ALLOW_ROOM_CREATE",
        "MATRIX_TOOLS_ALLOW_CROSS_ROOM",
        "MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    tokens = set_session_vars(platform="matrix", chat_id="!room:example.org", user_id="@alice:example.org")
    adapter = types.SimpleNamespace(
        _send_reaction=AsyncMock(return_value="$reaction"),
        redact_message=AsyncMock(return_value=True),
        create_room=AsyncMock(return_value="!new:example.org"),
        invite_user=AsyncMock(return_value=True),
        fetch_history=AsyncMock(return_value=[{"event_id": "$evt", "body": "hello"}]),
        set_presence=AsyncMock(return_value=True),
    )
    runner = types.SimpleNamespace(adapters={Platform.MATRIX: adapter})
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    try:
        yield adapter
    finally:
        for token in reversed(tokens):
            token.var.reset(token)


def _loads(result):
    return json.loads(result)


def test_matrix_tools_are_context_gated():
    from gateway.session_context import set_session_vars
    from tools.matrix_tools import matrix_set_presence

    tokens = set_session_vars(platform="discord", chat_id="123")
    try:
        result = _loads(matrix_set_presence({"state": "online"}))
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    assert "error" in result
    assert "only available" in result["error"]


def test_matrix_send_reaction_defaults_to_current_room(matrix_tool_context):
    from tools.matrix_tools import matrix_send_reaction

    result = _loads(matrix_send_reaction({"event_id": "$evt", "emoji": "👍"}))

    assert result == {"success": True, "event_id": "$reaction"}
    matrix_tool_context._send_reaction.assert_awaited_once_with(
        "!room:example.org", "$evt", "👍"
    )


def test_matrix_redact_message(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_redact_message

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
    result = _loads(matrix_redact_message({"event_id": "$evt", "reason": "cleanup"}))


    assert result == {"success": True}
    matrix_tool_context.redact_message.assert_awaited_once_with(
        "!room:example.org", "$evt", "cleanup"
    )


def test_matrix_redact_message_blocked_by_default(matrix_tool_context):
    from tools.matrix_tools import matrix_redact_message

    result = _loads(matrix_redact_message({"event_id": "$evt", "reason": "cleanup"}))

    assert "error" in result
    assert "MATRIX_TOOLS_ALLOW_REDACTION" in result["error"]
    matrix_tool_context.redact_message.assert_not_called()


def test_matrix_create_room_blocks_public_by_default(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_create_room

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "true")
    result = _loads(matrix_create_room({"name": "Public", "preset": "public_chat"}))

    assert "error" in result
    assert "MATRIX_ALLOW_PUBLIC_ROOMS" in result["error"]
    matrix_tool_context.create_room.assert_not_called()


def test_matrix_create_room_private(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_create_room

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", "true")
    result = _loads(matrix_create_room({"name": "Project", "invite": ["@bob:example.org"]}))

    assert result == {"success": True, "room_id": "!new:example.org"}
    matrix_tool_context.create_room.assert_awaited_once_with(
        name="Project",
        topic="",
        invite=["@bob:example.org"],
        is_direct=False,
        preset="private_chat",
    )


def test_matrix_create_room_blocked_by_default(matrix_tool_context):
    from tools.matrix_tools import matrix_create_room

    result = _loads(matrix_create_room({"name": "Project"}))

    assert "error" in result
    assert "MATRIX_TOOLS_ALLOW_ROOM_CREATE" in result["error"]
    matrix_tool_context.create_room.assert_not_called()


def test_matrix_invite_user(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_invite_user

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_INVITES", "true")
    result = _loads(matrix_invite_user({"user_id": "@bob:example.org"}))

    assert result == {"success": True}
    matrix_tool_context.invite_user.assert_awaited_once_with(
        "!room:example.org", "@bob:example.org"
    )


def test_matrix_invite_user_blocked_by_default(matrix_tool_context):
    from tools.matrix_tools import matrix_invite_user

    result = _loads(matrix_invite_user({"user_id": "@bob:example.org"}))

    assert "error" in result
    assert "MATRIX_TOOLS_ALLOW_INVITES" in result["error"]
    matrix_tool_context.invite_user.assert_not_called()


def test_matrix_fetch_history(matrix_tool_context):
    from tools.matrix_tools import matrix_fetch_history

    result = _loads(matrix_fetch_history({"limit": 5}))

    assert result == {"success": True, "events": [{"event_id": "$evt", "body": "hello"}]}
    matrix_tool_context.fetch_history.assert_awaited_once_with("!room:example.org", 5, "")


def test_matrix_tool_blocks_cross_room_reaction_by_default(matrix_tool_context):
    from tools.matrix_tools import matrix_send_reaction

    result = _loads(
        matrix_send_reaction(
            {"room_id": "!other:example.org", "event_id": "$evt", "emoji": "👍"}
        )
    )

    assert "error" in result
    assert "current room" in result["error"]
    matrix_tool_context._send_reaction.assert_not_called()


def test_matrix_tool_blocks_cross_room_history_by_default(matrix_tool_context):
    from tools.matrix_tools import matrix_fetch_history

    result = _loads(matrix_fetch_history({"room_id": "!other:example.org"}))

    assert "error" in result
    assert "current room" in result["error"]
    matrix_tool_context.fetch_history.assert_not_called()


def test_matrix_tool_blocks_cross_room_redaction_by_default(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_redact_message

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")
    result = _loads(
        matrix_redact_message({"room_id": "!other:example.org", "event_id": "$evt"})
    )

    assert "error" in result
    assert "current room" in result["error"]
    matrix_tool_context.redact_message.assert_not_called()


def test_matrix_tool_allows_cross_room_when_explicitly_enabled(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_send_reaction

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", "true")

    result = _loads(
        matrix_send_reaction(
            {"room_id": "!other:example.org", "event_id": "$evt", "emoji": "👍"}
        )
    )

    assert result == {"success": True, "event_id": "$reaction"}
    matrix_tool_context._send_reaction.assert_awaited_once_with(
        "!other:example.org", "$evt", "👍"
    )


def test_matrix_tool_cross_room_respects_allowed_rooms(matrix_tool_context, monkeypatch):
    from tools.matrix_tools import matrix_fetch_history

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", "true")
    monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!room:example.org")

    result = _loads(matrix_fetch_history({"room_id": "!other:example.org"}))

    assert "error" in result
    assert "MATRIX_ALLOWED_ROOMS" in result["error"]
    matrix_tool_context.fetch_history.assert_not_called()


def test_matrix_tool_cross_room_redaction_requires_destructive_opt_in(
    matrix_tool_context,
    monkeypatch,
):
    from tools.matrix_tools import matrix_redact_message

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", "true")
    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")

    result = _loads(
        matrix_redact_message({"room_id": "!other:example.org", "event_id": "$evt"})
    )

    assert "error" in result
    assert "DESTRUCTIVE" in result["error"]
    matrix_tool_context.redact_message.assert_not_called()


def test_matrix_tool_cross_room_redaction_allows_destructive_opt_in(
    matrix_tool_context,
    monkeypatch,
):
    from tools.matrix_tools import matrix_redact_message

    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", "true")
    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE", "true")
    monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", "true")

    result = _loads(
        matrix_redact_message({"room_id": "!other:example.org", "event_id": "$evt"})
    )

    assert result == {"success": True}
    matrix_tool_context.redact_message.assert_awaited_once_with(
        "!other:example.org", "$evt", ""
    )


def test_matrix_fetch_history_clamps_limit(matrix_tool_context):
    from tools.matrix_tools import matrix_fetch_history

    low = _loads(matrix_fetch_history({"limit": -5}))
    high = _loads(matrix_fetch_history({"limit": 1000}))

    assert low["success"] is True
    assert high["success"] is True
    assert matrix_tool_context.fetch_history.await_args_list[0].args == (
        "!room:example.org",
        1,
        "",
    )
    assert matrix_tool_context.fetch_history.await_args_list[1].args == (
        "!room:example.org",
        100,
        "",
    )


def test_matrix_set_presence(matrix_tool_context):
    from tools.matrix_tools import matrix_set_presence

    result = _loads(matrix_set_presence({"state": "unavailable", "status_msg": "busy"}))

    assert result == {"success": True}
    matrix_tool_context.set_presence.assert_awaited_once_with("unavailable", "busy")
