"""Regression tests for the send_message silent home-channel leak.

Bug (root-caused 2026-06-21): a ``send_message`` / ``react`` call with NO
explicit chat target silently fell back to the platform's *global home channel*.
An agent posting a mid-turn update from a non-home channel leaked the message
into the home channel (sometimes a different platform's home / a DM).

These tests exercise the real ``_handle_send`` / ``_handle_react`` path with the
real session-context ContextVars the gateway binds per turn; only the network
send (``_send_to_platform``) and the live config are stubbed. They assert:

  * AC1  — bare send inside a live messaging turn → the turn's OWN channel.
  * AC1b — bare react inside a live messaging turn → the turn's OWN channel.
  * AC2  — bare send with no messaging context (cron/CLI) → home (unchanged).
  * AC2b — bare send with HERMES_GATEWAY_SESSION set but NO platform bound
           (TUI/desktop/ACP) → home, NOT an error (Pass-2 blocker guard).
  * AC3  — bare send under HERMES_CRON_SESSION (+synthetic platform) → home.
  * AC4  — bare send in a turn whose origin can't be resolved → error, NOT home.
  * AC5  — explicit cross-channel target unchanged.
  * AC7b — origin thread id is preserved.
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from gateway.config import Platform  # noqa: E402
from gateway.session_context import clear_session_vars, set_session_vars  # noqa: E402
from tools.send_message_tool import send_message_tool  # noqa: E402

HOME_CHAT = "home-9999"
ORIGIN_CHAT = "origin-1515"
ORIGIN_THREAD = "thr-42"


def _run_async_immediately(coro):
    import asyncio

    return asyncio.run(coro)


def _config_with_home(home_chat=HOME_CHAT):
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    discord_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    home = SimpleNamespace(chat_id=home_chat, name="Home")
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg, Platform.DISCORD: discord_cfg},
        get_home_channel=lambda _platform: home,
    )


@pytest.fixture
def _clear_session():
    """Ensure no leftover session/cron context bleeds between tests."""
    for var in (
        "HERMES_CRON_SESSION",
        "HERMES_GATEWAY_SESSION",
        "HERMES_CRON_AUTO_DELIVER_PLATFORM",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    ):
        os.environ.pop(var, None)
    tokens = set_session_vars(platform="", chat_id="", chat_name="", thread_id="")
    yield
    clear_session_vars(tokens)


def _send(target, message="checkpoint"):
    return json.loads(
        send_message_tool({"action": "send", "target": target, "message": message})
    )


# --------------------------------------------------------------------------- #
# AC1 — bare send inside a live messaging turn -> the turn's OWN channel
# --------------------------------------------------------------------------- #
def test_bare_target_in_turn_routes_to_origin(_clear_session):
    """The exact leak: a bare 'discord' send during a live Discord turn must
    land in the turn's channel, never the global home channel."""
    config = _config_with_home()
    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("discord")
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    sent_chat = send_mock.await_args.args[2]
    assert sent_chat == ORIGIN_CHAT, f"leaked to {sent_chat!r}, expected origin"
    assert sent_chat != HOME_CHAT


# --------------------------------------------------------------------------- #
# AC1b — bare react inside a live messaging turn -> the turn's OWN channel
# --------------------------------------------------------------------------- #
def test_bare_react_in_turn_routes_to_origin(_clear_session):
    config = _config_with_home()
    captured = {}

    class _Adapter:
        async def add_reaction(self, chat_id=None, emoji=None, message_id=None):
            captured["chat_id"] = chat_id
            return {"success": True}

    runner = SimpleNamespace(adapters={Platform.DISCORD: _Adapter()})
    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("gateway.run._gateway_runner_ref", return_value=runner), \
             patch("model_tools._run_async", side_effect=_run_async_immediately):
            result = json.loads(
                send_message_tool(
                    {"action": "react", "target": "discord", "emoji": "✅"}
                )
            )
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert captured["chat_id"] == ORIGIN_CHAT
    assert captured["chat_id"] != HOME_CHAT


# --------------------------------------------------------------------------- #
# AC2 — no messaging context (cron/CLI) -> home (unchanged)
# --------------------------------------------------------------------------- #
def test_bare_target_no_context_routes_to_home(_clear_session):
    config = _config_with_home()
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _send("discord")

    assert result["success"] is True
    assert send_mock.await_args.args[2] == HOME_CHAT


# --------------------------------------------------------------------------- #
# AC2b — HERMES_GATEWAY_SESSION set but NO platform bound (TUI/desktop/ACP)
#        -> home, NOT an error. (Pass-2 blocker regression guard.)
# --------------------------------------------------------------------------- #
def test_bare_target_gateway_session_no_platform_routes_to_home(_clear_session):
    config = _config_with_home()
    os.environ["HERMES_GATEWAY_SESSION"] = "1"
    # TUI binds only a session_key, no messaging platform.
    tokens = set_session_vars(session_key="some-tui-uuid")
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("discord")
    finally:
        clear_session_vars(tokens)
        os.environ.pop("HERMES_GATEWAY_SESSION", None)

    assert "error" not in result, f"TUI bare send wrongly errored: {result}"
    assert result["success"] is True
    assert send_mock.await_args.args[2] == HOME_CHAT


# --------------------------------------------------------------------------- #
# AC3 — cron (+ synthetic platform) -> home, not errored
# --------------------------------------------------------------------------- #
def test_bare_target_cron_routes_to_home(_clear_session):
    config = _config_with_home()
    os.environ["HERMES_CRON_SESSION"] = "1"
    # Real cron clears the platform, but synthetically bind one to prove the
    # HERMES_CRON_SESSION leg of _has_messaging_origin is the load-bearing guard.
    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("discord")
    finally:
        clear_session_vars(tokens)
        os.environ.pop("HERMES_CRON_SESSION", None)

    assert "error" not in result
    assert result["success"] is True
    assert send_mock.await_args.args[2] == HOME_CHAT


# --------------------------------------------------------------------------- #
# AC4 — live turn, origin unresolvable for the requested platform -> error
# --------------------------------------------------------------------------- #
def test_in_turn_unresolvable_origin_errors_not_home(_clear_session):
    """Turn is on discord; agent does a bare send to telegram (a different
    platform, no chat) -> must error, never silently hit telegram's home."""
    config = _config_with_home()
    send_mock = AsyncMock(return_value={"success": True})
    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("telegram")
    finally:
        clear_session_vars(tokens)

    assert "error" in result
    assert "could not be resolved" in result["error"]
    send_mock.assert_not_awaited()


# --------------------------------------------------------------------------- #
# AC5 — explicit cross-channel target unchanged
# --------------------------------------------------------------------------- #
def test_explicit_target_unchanged(_clear_session):
    config = _config_with_home()
    # Even inside a discord turn, an explicit chat id is honored verbatim.
    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("discord:7777000011112222")
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert send_mock.await_args.args[2] == "7777000011112222"


# --------------------------------------------------------------------------- #
# AC7 — origin resolution reads task-local session-context (not a global)
# --------------------------------------------------------------------------- #
def test_origin_resolution_reads_session_context(_clear_session):
    from tools.send_message_tool import _resolve_send_target, _SEND_TARGET_ORIGIN

    tokens = set_session_vars(platform="discord", chat_id=ORIGIN_CHAT)
    try:
        state, chat_id, _thread = _resolve_send_target("discord")
    finally:
        clear_session_vars(tokens)
    assert state == _SEND_TARGET_ORIGIN
    assert chat_id == ORIGIN_CHAT

    # With nothing bound, it's NOT_IN_TURN (home path).
    from tools.send_message_tool import _SEND_TARGET_NOT_IN_TURN
    state2, _c, _t = _resolve_send_target("discord")
    assert state2 == _SEND_TARGET_NOT_IN_TURN


# --------------------------------------------------------------------------- #
# AC7b — origin thread id is preserved
# --------------------------------------------------------------------------- #
def test_origin_preserves_thread_id(_clear_session):
    config = _config_with_home()
    tokens = set_session_vars(
        platform="discord", chat_id=ORIGIN_CHAT, thread_id=ORIGIN_THREAD
    )
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = _send("discord")
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert send_mock.await_args.args[2] == ORIGIN_CHAT
    assert send_mock.await_args.kwargs.get("thread_id") == ORIGIN_THREAD
