"""Regression tests for #126732: Telegram DM-topic sessions surface in Desktop Bot Mode.

A profile serving a Telegram gateway with DM topic mode holds one session per
DM topic (same chat_id, different thread_id — lane isolation). Those rows are
healthy (titled, hidden=0, archived=0) yet were invisible in Bot Mode: the bot
row opens the canonical Bot Chat (by name, never recency), and the Sessions
sidebar fetched/filtered the ambient gateway scope instead of the selected
bot's profile, so the bot's Telegram threads had no door.

The fix scopes the sidebar fetch + display to the selected bot's profile when
the workspace owns a bot (Bot Mode), and resolves every recents-only session
lookup across all slices (recents + cron + messaging) so Telegram rows
(messaging slice only) stay findable/renamable/deletable.

What this locks in (and what it must not break):
- Messaging slice includes Telegram DM-topic threads (titled, visible).
- Profiles roster reports the newest visible thread as last_session (the
  "Open recent session" door); canonical_session stays the Bot Chat.
- Lane isolation: same chat_id with different thread_id are different
  sessions; same thread_id is the same session (#67945).
- Gateway origin scoping still blocks cross-participant resume/listing
  (_resume_row_visible / _resume_target_allowed); Desktop profile-scoped
  listing may enumerate a profile's own threads (operator owns the bot).
- hidden=true still marks Bot Mode plumbing only; gateway threads stay
  hidden=0 and visible (no flag reuse).
"""

from __future__ import annotations

import time

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _make_source(chat_id="123", thread_id="2", user_id="123", chat_type="dm"):
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id=str(chat_id), chat_type=chat_type,
        user_id=str(user_id), thread_id=str(thread_id),
    )


def _add_telegram_thread(db, sid, *, thread_id, title, ts, chat_id="123", user_id="123"):
    seed_session(
        db, sid, "telegram", title=title, user_id=str(user_id), chat_id=str(chat_id),
        chat_type="dm", thread_id=str(thread_id),
        session_key=f"agent:mybot:telegram:dm:{chat_id}:{thread_id}",
    )
    seed_messages(db, sid, {"role": "user", "content": f"hello in {title}", "timestamp": ts},
                  {"role": "assistant", "content": "hi back", "timestamp": ts + 1})


def _add_bot_chat(db, sid="botchat1", ts=None):
    ts = ts if ts is not None else time.time() - 100
    seed_session(db, sid, "desktop", title="Bot Chat")
    seed_messages(db, sid, {"role": "user", "content": "hello bot", "timestamp": ts})
    db.sessions.set_hidden(sid, True)


def _list_rows(db, limit, **filters):
    """The listing rows, most recently active first, as the Desktop session list reads them."""
    from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow

    entries = db.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE, **filters),
                                        SessionWindow(limit=limit), compact_rows=True)
    return [entry.as_list_row() for entry in entries]


def test_telegram_dm_topic_threads_are_distinct_sessions(tmp_path):
    """Lane isolation: different thread_id in one DM are different sessions."""
    db = storage_handle(tmp_path)
    now = time.time()
    _add_telegram_thread(db, "tg2", thread_id="2", title="Topic A", ts=now - 20)
    _add_telegram_thread(db, "tg3", thread_id="3", title="Topic B", ts=now - 10)

    rows = {s["id"]: s for s in _list_rows(db, 20, include_hidden=True)}
    assert rows["tg2"]["thread_id"] == "2"
    assert rows["tg3"]["thread_id"] == "3"
    assert rows["tg2"]["title"] == "Topic A"
    assert rows["tg3"]["title"] == "Topic B"
    # Same chat, different threads → different rows (not merged).
    assert rows["tg2"]["chat_id"] == rows["tg3"]["chat_id"] == "123"
    assert rows["tg2"]["id"] != rows["tg3"]["id"]


def test_telegram_threads_surface_in_messaging_slice_not_recents(tmp_path):
    """Desktop sidebar split: telegram threads belong to messaging, never recents."""
    db = storage_handle(tmp_path)
    now = time.time()
    _add_bot_chat(db, ts=now - 100)
    _add_telegram_thread(db, "tg1", thread_id="2", title="Telegram chat", ts=now - 10)

    messaging_exclude = ("cron", "acp", "cli", "codex", "desktop", "gateway",
                         "kanban", "local", "oneshot", "tui")
    messaging = _list_rows(db, 100, exclude_sources=messaging_exclude, min_message_count=1,
                           include_archived=False, archived_only=False, include_pinned=True)
    assert [s["id"] for s in messaging] == ["tg1"]
    assert messaging[0]["hidden"] == 0
    assert messaging[0]["archived"] == 0
    assert messaging[0]["title"] == "Telegram chat"

    recents_exclude = ("acp", "cron", "kanban", "oneshot", "subagent", "tool",
                       "telegram", "discord", "slack", "mattermost", "matrix",
                       "signal", "whatsapp", "bluebubbles", "photon",
                       "homeassistant", "email", "sms", "webhook", "api_server",
                       "weixin", "wecom", "qqbot", "yuanbao", "dingtalk", "feishu")
    recents = _list_rows(db, 20, exclude_sources=recents_exclude, min_message_count=1,
                         include_archived=False, archived_only=False, include_pinned=True)
    # Bot Chat is hidden (canonical, never in recents); telegram lives in
    # messaging, so recents is empty here — not missing, just split.
    assert [s["id"] for s in recents] == []


def test_profiles_roster_reports_telegram_as_last_session(tmp_path):
    """Roster door: last_session is the newest visible thread (telegram when
    freshest); canonical_session stays the hidden Bot Chat."""
    from tui_gateway.methods_session import _denied_source
    from tui_gateway.methods_profiles import _canonical_session_row, _latest_profile_session_rows

    db = storage_handle(tmp_path)
    now = time.time()
    _add_bot_chat(db, ts=now - 100)
    _add_telegram_thread(db, "tg1", thread_id="2", title="Telegram chat", ts=now - 10)

    # Mirror _latest_profile_session_rows (deny-list + newest-visible wins).
    rows = _list_rows(db, 20)
    human = next((s for s in rows if not _denied_source(s)), None)
    assert human is not None and human["id"] == "tg1"

    # Mirror _canonical_session_row (exact-title registry, hidden resolves).
    record = db.sessions.get_by_title("Bot Chat")
    row = record.as_session_row() if record is not None else None
    assert row is not None and row["id"] == "botchat1"
    assert row["hidden"] == 1


def test_gateway_origin_scoping_still_blocks_cross_topic_and_cross_user():
    """Guard rails the fix must not break: a group thread is a different session
    even with the same chat_id, and a persisted row only proves its own lane
    (IDOR scoping preserved). Topic lanes of one PRIVATE chat are the same
    person's DM and may share /resume by design (#57821)."""
    from gateway.slash_commands_session import GatewaySessionCommandsMixin as Mixin

    cur = _make_source(thread_id="2")
    same_thread = _make_source(thread_id="2")
    group_thread = _make_source(chat_id="-100", thread_id="2", chat_type="group")
    other_group_thread = _make_source(chat_id="-100", thread_id="3", chat_type="group")
    other_user = _make_source(thread_id="2", user_id="999", chat_id="999")

    assert Mixin._same_origin_chat(Mixin, cur, same_thread) is True
    assert Mixin._same_origin_chat(Mixin, group_thread, other_group_thread) is False
    assert Mixin._same_origin_chat(Mixin, cur, other_user) is False
    assert Mixin._same_origin_chat(Mixin, cur, None) is False
