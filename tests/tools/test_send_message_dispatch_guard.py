"""Tests for the send_message dispatch guard (v7).

Covers the ``_validate_dispatch_target`` helper that prevents group-triggered
agent dispatches from being misrouted to a user's DM.
"""

import os

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools.send_message_tool import _validate_dispatch_target


GROUP_CHAT_ID = "-1003732657330"
BOSS_USER_ID = "6590198078"
OTHER_GROUP_CHAT_ID = "-1009999999999"
STRANGER_USER_ID = "1234567890"


@pytest.fixture
def tg_group_session():
    """Session context: triggered by the boss in the main group."""
    tokens = set_session_vars(
        platform="telegram",
        chat_id=GROUP_CHAT_ID,
        user_id=BOSS_USER_ID,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def tg_dm_session():
    """Session context: triggered by the boss in a direct message."""
    tokens = set_session_vars(
        platform="telegram",
        chat_id=BOSS_USER_ID,
        user_id=BOSS_USER_ID,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def test_group_trigger_dm_target_with_bot_mention_redirects(tg_group_session):
    """The real-world bug: group trigger → DM target + @bot mention → redirect."""
    new_chat, warning = _validate_dispatch_target(
        "telegram",
        BOSS_USER_ID,
        '@hm_xiaoxiong_content_bot 写一段 200 字 SEO 文案',
    )
    assert new_chat == GROUP_CHAT_ID
    assert warning is not None
    assert "auto-redirected" in warning


def test_group_trigger_dm_target_is_trigger_user_redirects_without_mention(tg_group_session):
    """Even without a bot mention, DM to the triggering user is a classic mix-up."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", BOSS_USER_ID, "just a note to boss"
    )
    assert new_chat == GROUP_CHAT_ID
    assert warning is not None


def test_group_trigger_same_group_target_no_redirect(tg_group_session):
    """Sending to the trigger group itself is always fine."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", GROUP_CHAT_ID, "@hm_xiaoxiong_content_bot 任务"
    )
    assert new_chat == GROUP_CHAT_ID
    assert warning is None


def test_group_trigger_different_group_target_no_redirect(tg_group_session):
    """Dispatching to another group is legitimate cross-chat usage."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", OTHER_GROUP_CHAT_ID, "@hm_xiaoxiong_content_bot 任务"
    )
    assert new_chat == OTHER_GROUP_CHAT_ID
    assert warning is None


def test_group_trigger_stranger_dm_no_mention_no_redirect(tg_group_session):
    """DM to someone who's not the trigger user, no @bot → legitimate, don't touch."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", STRANGER_USER_ID, "hello friend"
    )
    assert new_chat == STRANGER_USER_ID
    assert warning is None


def test_group_trigger_stranger_dm_with_bot_mention_redirects(tg_group_session):
    """@bot mention alone signals dispatch intent — redirect even for stranger DM."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", STRANGER_USER_ID, "@hm_xiaoxiong_content_bot 写文案"
    )
    assert new_chat == GROUP_CHAT_ID
    assert warning is not None


def test_dm_trigger_any_target_no_redirect(tg_dm_session):
    """When agent is triggered in a DM, don't second-guess its targets."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", BOSS_USER_ID, "@hm_xiaoxiong_content_bot do X"
    )
    assert new_chat == BOSS_USER_ID
    assert warning is None


def test_no_session_context_no_redirect():
    """CLI/cron without gateway session vars → guard is a no-op."""
    # Ensure no stale env vars leak in from the runner
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_USER_ID",
    ):
        os.environ.pop(key, None)

    new_chat, warning = _validate_dispatch_target(
        "telegram", BOSS_USER_ID, "@hm_xiaoxiong_content_bot do X"
    )
    assert new_chat == BOSS_USER_ID
    assert warning is None


def test_non_telegram_platform_no_redirect(tg_group_session):
    """Guard is Telegram-only for now."""
    new_chat, warning = _validate_dispatch_target(
        "discord", BOSS_USER_ID, "@hm_xiaoxiong_content_bot do X"
    )
    assert new_chat == BOSS_USER_ID
    assert warning is None


def test_none_chat_id_returns_as_is(tg_group_session):
    """Unresolved chat_id (None) passes through unchanged."""
    new_chat, warning = _validate_dispatch_target("telegram", None, "any message")
    assert new_chat is None
    assert warning is None


def test_non_numeric_chat_id_passes_through(tg_group_session):
    """Non-numeric chat_id (e.g. channel name left unresolved) shouldn't crash."""
    new_chat, warning = _validate_dispatch_target(
        "telegram", "not-a-number", "@hm_xiaoxiong_content_bot"
    )
    assert new_chat == "not-a-number"
    assert warning is None
