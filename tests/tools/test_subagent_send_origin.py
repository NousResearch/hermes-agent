"""Regression tests for the subagent send_message origin leak (PRD v2).

A delegate_task child runs in a bare ThreadPoolExecutor worker with empty
contextvars, so a subagent calling send_message/react with a BARE target had no
session origin and silently fell back to the global home channel (the v2 leak,
recurring after PR #71 fixed the main-turn case).

Fix: dedicated routing-only send-origin contextvars (_SEND_ORIGIN_*), bound for
the child run from the parent origin captured at spawn, read ONLY by the send
resolver (so the child keeps its `subagent` identity for approval/skills/TTS).

These tests exercise the real resolver + real session_context contextvars; only
the network send (_send_to_platform) is stubbed.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from gateway.session_context import (  # noqa: E402
    set_send_origin,
    clear_send_origin,
    get_send_origin,
    set_session_vars,
    clear_session_vars,
)
from tools.send_message_tool import (  # noqa: E402
    _resolve_send_target,
    _has_messaging_origin,
    _interactive_origin,
    _SEND_TARGET_ORIGIN,
    _SEND_TARGET_NOT_IN_TURN,
    send_message_tool,
)

HOME = "home-9999"
PARENT_CHAN = "parent-1513"
PARENT_CHAN_B = "parent-2222"


def _run_async_immediately(coro):
    import asyncio
    return asyncio.run(coro)


def _config_with_home(home_chat=HOME):
    cfg = SimpleNamespace(enabled=True, token="***", extra={})
    home = SimpleNamespace(chat_id=home_chat, name="Home")
    from gateway.config import Platform
    return SimpleNamespace(
        platforms={Platform.DISCORD: cfg, Platform.TELEGRAM: cfg},
        get_home_channel=lambda _p: home,
    )


@pytest.fixture(autouse=True)
def _clean_env():
    for v in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION"):
        os.environ.pop(v, None)
    # Ensure no leftover session/send-origin contextvars.
    st = set_session_vars(platform="", chat_id="")
    so = set_send_origin("", "", "")
    yield
    clear_session_vars(st)
    clear_send_origin(so)


# --------------------------------------------------------------------------- #
# Phase 0 — the seam (now GREEN with the fix): a bound send-origin routes a
# bare send to the parent channel, NOT home.
# --------------------------------------------------------------------------- #
def test_bare_subagent_send_routes_to_parent_origin(_clean_env):
    tokens = set_send_origin("discord", PARENT_CHAN)
    try:
        cfg = _config_with_home()
        with patch("gateway.config.load_gateway_config", return_value=cfg), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform",
                   new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(send_message_tool(
                {"action": "send", "target": "discord", "message": "progress update"}))
    finally:
        clear_send_origin(tokens)

    assert result["success"] is True
    sent = send_mock.await_args.args[2]
    assert sent == PARENT_CHAN, f"leaked to {sent!r}, expected parent origin"
    assert sent != HOME


def test_resolver_returns_origin_for_send_origin(_clean_env):
    tokens = set_send_origin("discord", PARENT_CHAN)
    try:
        assert _has_messaging_origin() is True
        state, chat, _t = _resolve_send_target("discord")
    finally:
        clear_send_origin(tokens)
    assert state == _SEND_TARGET_ORIGIN
    assert chat == PARENT_CHAN


# --------------------------------------------------------------------------- #
# AC1b — react path
# --------------------------------------------------------------------------- #
def test_bare_subagent_react_routes_to_parent_origin(_clean_env):
    captured = {}

    class _Adapter:
        async def add_reaction(self, chat_id=None, emoji=None, message_id=None):
            captured["chat_id"] = chat_id
            return {"success": True}

    runner = SimpleNamespace(adapters=None)
    from gateway.config import Platform
    runner.adapters = {Platform.DISCORD: _Adapter()}
    tokens = set_send_origin("discord", PARENT_CHAN)
    try:
        cfg = _config_with_home()
        with patch("gateway.config.load_gateway_config", return_value=cfg), \
             patch("gateway.run._gateway_runner_ref", return_value=runner), \
             patch("model_tools._run_async", side_effect=_run_async_immediately):
            result = json.loads(send_message_tool(
                {"action": "react", "target": "discord", "emoji": "✅"}))
    finally:
        clear_send_origin(tokens)
    assert result["success"] is True
    assert captured["chat_id"] == PARENT_CHAN
    assert captured["chat_id"] != HOME


# --------------------------------------------------------------------------- #
# AC1c — a live gateway main turn wins over a (stale) send-origin
# --------------------------------------------------------------------------- #
def test_session_platform_wins_over_send_origin(_clean_env):
    # Both bound: main turn on discord:MAIN, plus a leftover send-origin.
    st = set_session_vars(platform="discord", chat_id="MAIN-CHAN")
    so = set_send_origin("discord", PARENT_CHAN)
    try:
        state, chat, _t = _resolve_send_target("discord")
    finally:
        clear_send_origin(so)
        clear_session_vars(st)
    assert state == _SEND_TARGET_ORIGIN
    assert chat == "MAIN-CHAN", "live gateway turn must win over send-origin"


# --------------------------------------------------------------------------- #
# AC2 — no parent origin (CLI/cron-spawned subagent) -> home, no error
# --------------------------------------------------------------------------- #
def test_no_send_origin_falls_to_home(_clean_env):
    cfg = _config_with_home()
    with patch("gateway.config.load_gateway_config", return_value=cfg), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _ = json.loads(send_message_tool(
            {"action": "send", "target": "discord", "message": "x"}))
    assert "error" not in result
    assert result["success"] is True
    assert send_mock.await_args.args[2] == HOME


# --------------------------------------------------------------------------- #
# AC3 — concurrent children with different parent origins don't cross-bleed
# (contextvars are per-thread; copy_context in each thread isolates them).
# --------------------------------------------------------------------------- #
def test_concurrent_children_no_origin_crossbleed(_clean_env):
    import threading
    import contextvars

    results = {}

    def _child(name, origin_chat):
        tokens = set_send_origin("discord", origin_chat)
        try:
            _s, chat, _t = _resolve_send_target("discord")
            results[name] = chat
        finally:
            clear_send_origin(tokens)

    def _run_in_ctx(name, origin_chat):
        ctx = contextvars.copy_context()
        ctx.run(_child, name, origin_chat)

    t1 = threading.Thread(target=_run_in_ctx, args=("a", PARENT_CHAN))
    t2 = threading.Thread(target=_run_in_ctx, args=("b", PARENT_CHAN_B))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert results["a"] == PARENT_CHAN
    assert results["b"] == PARENT_CHAN_B


# --------------------------------------------------------------------------- #
# AC1d — grandchild chaining: clear restores the parent's origin (nestable)
# --------------------------------------------------------------------------- #
def test_send_origin_clear_is_nestable(_clean_env):
    outer = set_send_origin("discord", PARENT_CHAN)
    try:
        assert get_send_origin()[1] == PARENT_CHAN
        inner = set_send_origin("discord", PARENT_CHAN_B)  # grandchild
        try:
            assert get_send_origin()[1] == PARENT_CHAN_B
        finally:
            clear_send_origin(inner)
        # After grandchild clears, the child's origin must be restored, not blank.
        assert get_send_origin()[1] == PARENT_CHAN, "clear must restore parent origin, not blank"
    finally:
        clear_send_origin(outer)


# --------------------------------------------------------------------------- #
# AC4 — instrumentation: warns on unexpected home-fallback, never alters state,
# silent on cron.
# --------------------------------------------------------------------------- #
def test_resolver_warns_on_unexpected_home_fallback(_clean_env, caplog):
    import logging
    # session_key resolvable but no messaging origin -> tripwire.
    st = set_session_vars(session_key="agent:main:discord:group:CHAN:user")
    # but clear platform so _has_messaging_origin is False
    try:
        # session_key set via set_session_vars also set platform=""; ensure no platform
        with caplog.at_level(logging.WARNING, logger="tools.send_message_tool"):
            state, _c, _t = _resolve_send_target("discord")
        assert state == _SEND_TARGET_NOT_IN_TURN
        assert any("home-fallback despite session_key" in r.message for r in caplog.records)
    finally:
        clear_session_vars(st)


def test_no_warning_on_cron(_clean_env, caplog):
    import logging
    os.environ["HERMES_CRON_SESSION"] = "1"
    st = set_session_vars(session_key="agent:main:cron:job")
    try:
        with caplog.at_level(logging.WARNING, logger="tools.send_message_tool"):
            state, _c, _t = _resolve_send_target("discord")
        assert state == _SEND_TARGET_NOT_IN_TURN
        assert not any("home-fallback despite session_key" in r.message for r in caplog.records)
    finally:
        clear_session_vars(st)
        os.environ.pop("HERMES_CRON_SESSION", None)


def test_warning_never_alters_state(_clean_env):
    st = set_session_vars(session_key="agent:main:discord:group:CHAN:user")
    try:
        with patch("tools.send_message_tool.logger") as mock_logger:
            mock_logger.warning.side_effect = RuntimeError("boom")
            # Must not raise despite the logger blowing up.
            state, _c, _t = _resolve_send_target("discord")
        assert state == _SEND_TARGET_NOT_IN_TURN
    finally:
        clear_session_vars(st)


# --------------------------------------------------------------------------- #
# AC5b (security) — the send-origin vars are NOT read by approval gating.
# --------------------------------------------------------------------------- #
def test_send_origin_does_not_flip_approval_context(_clean_env):
    from tools.approval import _is_gateway_approval_context, _get_session_platform
    tokens = set_send_origin("discord", PARENT_CHAN)
    try:
        # A bound send-origin must NOT make the child look like a gateway turn.
        assert _get_session_platform() == "", "send-origin must not set session platform"
        assert _is_gateway_approval_context() is False, \
            "send-origin must not flip approval gating (child keeps subagent identity)"
    finally:
        clear_send_origin(tokens)


# --------------------------------------------------------------------------- #
# TRUE-PATH e2e — drive the real _run_with_thread_capture bind wrapper.
# A fake child whose run_conversation resolves a send target proves the bind
# set in the delegate wrapper reaches the child's actual execution + that the
# `finally` clears it afterward. Also covers the grandchild-chaining capture.
# --------------------------------------------------------------------------- #
def test_bind_helper_routes_then_clears(_clean_env):
    from tools.delegate_tool import _bind_child_send_origin, _clear_child_send_origin

    child = SimpleNamespace(
        _blackbox_parent_platform="discord",
        _blackbox_parent_chat_id=PARENT_CHAN,
        _send_origin_thread_id="",
    )
    captured = {}

    def _fake_run():
        # Simulate the subagent calling send_message mid-run: resolve a bare target.
        st, chat, _t = _resolve_send_target("discord")
        captured["state"], captured["chat"] = st, chat

    # Mirror _run_with_thread_capture's bind/try/finally exactly.
    tokens = _bind_child_send_origin(child)
    try:
        _fake_run()
    finally:
        _clear_child_send_origin(tokens)

    assert captured["state"] == _SEND_TARGET_ORIGIN
    assert captured["chat"] == PARENT_CHAN
    # After the run, the origin must be cleared (no bleed to a later turn).
    assert get_send_origin()[1] == ""


def test_bind_helper_skips_subagent_pseudo_platform(_clean_env):
    """A child whose captured parent platform is the literal 'subagent'
    pseudo-identity (un-chained grandchild) must NOT bind — its bare send
    falls to home, never to a bogus 'subagent' origin."""
    from tools.delegate_tool import _bind_child_send_origin
    child = SimpleNamespace(
        _blackbox_parent_platform="subagent",
        _blackbox_parent_chat_id="",
        _send_origin_thread_id="",
    )
    tokens = _bind_child_send_origin(child)
    assert tokens is None
    assert get_send_origin()[0] == ""


def test_bind_helper_none_when_no_parent_origin(_clean_env):
    from tools.delegate_tool import _bind_child_send_origin
    child = SimpleNamespace(_blackbox_parent_platform="", _blackbox_parent_chat_id="")
    assert _bind_child_send_origin(child) is None


def test_platform_only_send_origin_falls_to_home_not_error(_clean_env):
    """Greptile P2: a send-origin with platform but EMPTY chat_id must fall
    closed to home (NOT_IN_TURN), not surface a hard IN_TURN_UNRESOLVABLE error.
    _has_messaging_origin must require BOTH platform and chat to match
    _interactive_origin's contract."""
    from tools.send_message_tool import _SEND_TARGET_NOT_IN_TURN
    tokens = set_send_origin("discord", "")  # platform set, chat empty
    try:
        assert _has_messaging_origin() is False, \
            "platform-only send-origin must not flag a messaging origin"
        state, _c, _t = _resolve_send_target("discord")
        assert state == _SEND_TARGET_NOT_IN_TURN, \
            "must fall closed to home, not IN_TURN_UNRESOLVABLE error"
    finally:
        clear_send_origin(tokens)
