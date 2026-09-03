"""Kanban auto-subscribe: unattached/background home-channel fallback.

Covers the root-cause fix for boards where agent/CLI/background-created
tasks never reached the configured home channels:

  - kanban_create without gateway/TUI origin context (CLI, cron,
    background agents, dispatcher-spawned workers) falls back to
    subscribing every ENABLED configured home channel when
    ``kanban.auto_subscribe_on_create`` is enabled.
  - Gateway/TUI-originated creates keep their origin subscription and do
    NOT get duplicate home rows.
  - Re-running subscribe-to-homes is idempotent (no duplicates).
  - A missing/broken homes config must not fail task creation.
  - Platforms with ``enabled: false`` never produce dead subscription rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers (same shape as the existing tool-surface tests)
# ---------------------------------------------------------------------------


def _list_subs_for_task(task_id):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        return list(kb.list_notify_subs(conn, task_id))
    finally:
        conn.close()


def _sub_keys(subs):
    return sorted((s["platform"], s["chat_id"], s.get("thread_id") or "") for s in subs)


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Same shape as tests/tools/test_kanban_tools.py::worker_env."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


@pytest.fixture
def with_home_channels_env(monkeypatch):
    """Simulate telegram + discord homes configured via env overlays."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:fake")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1234567")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "9999999")


def _clear_session_env(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)


def test_unattached_create_subscribes_configured_home_channels(
    monkeypatch,
    worker_env,
    with_home_channels_env,
):
    """A kanban_create from a session WITHOUT gateway/TUI delivery context
    (CLI / cron / background agent / dispatcher-spawned worker) still lands
    in the configured home channels so orchestration tasks notify."""
    from tools import kanban_tools as kt

    _clear_session_env(monkeypatch)

    out = kt._handle_create({
        "title": "background orchestration task",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d.get("subscribed") is True, (
        "unattached create should report subscribed=True via home fallback"
    )

    subs = _list_subs_for_task(d["task_id"])
    assert _sub_keys(subs) == [
        ("discord", "9999999", ""),
        ("telegram", "1234567", "42"),
    ]
    # Home-fallback rows carry the active profile so THIS profile's gateway
    # notifier owns delivery (the ownership gate drops foreign-profile rows).
    assert {s["notifier_profile"] for s in subs} == {"test-worker"}


def test_unattached_create_homes_disabled_by_config_gate(
    monkeypatch,
    worker_env,
    tmp_path,
    with_home_channels_env,
):
    """kanban.auto_subscribe_on_create=false suppresses the home fallback too —
    same knob as the gateway/TUI path (explicit opt-out)."""
    home = tmp_path / "gate-home" / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "gateway:\n  enabled: false\nkanban:\n  auto_subscribe_on_create: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    _clear_session_env(monkeypatch)

    from tools import kanban_tools as kt

    out = kt._handle_create({
        "title": "gated off",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d.get("subscribed") is False
    assert _list_subs_for_task(d["task_id"]) == []


def test_disabled_platform_home_is_skipped_no_dead_rows(
    monkeypatch,
    worker_env,
    tmp_path,
):
    """Security finding: a platform configured ``enabled: false`` has no
    adapter to deliver through — its home must NOT produce a dead
    subscription row that the notifier ownership gate would skip forever."""
    import yaml

    from hermes_cli import kanban_db as kb

    # Isolate config entirely from the shared env-overlay fixtures: the
    # enabled/disabled split lives in config.yaml alone here.
    for var in (
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_THREAD_ID", "DISCORD_BOT_TOKEN",
        "DISCORD_HOME_CHANNEL", "SLACK_BOT_TOKEN", "SLACK_HOME_CHANNEL",
    ):
        monkeypatch.delenv(var, raising=False)

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "gateway": {"enabled": False},
            "platforms": {
                "telegram": {
                    "enabled": True,
                    "home_channel": {
                        "platform": "telegram",
                        "chat_id": "1234567",
                        "name": "Home",
                        "thread_id": "42",
                    },
                },
                "discord": {
                    "enabled": False,
                    "home_channel": {
                        "platform": "discord",
                        "chat_id": "9999999",
                        "name": "Home",
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    homes = kb.configured_home_channels()
    # The invariant under test: disabled platform → no dead row.
    assert homes == [
        {"platform": "telegram", "chat_id": "1234567", "thread_id": "42", "name": "Home"},
    ]

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="no dead rows", assignee="w1")
        kb.subscribe_task_to_configured_homes(conn, tid)
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert {s["platform"] for s in subs} == {"telegram"}


def test_gateway_origin_create_does_not_duplicate_home_subscription(
    monkeypatch,
    worker_env,
    with_home_channels_env,
):
    """A gateway-session create keeps its ORIGIN subscription. The home
    fallback must not add a second row for the same platform when the
    origin already covers it — no duplicate notifications for the chat
    that created the task."""
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-7")

    out = kt._handle_create({
        "title": "origin wins over home",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is True

    subs = _list_subs_for_task(d["task_id"])
    keys = _sub_keys(subs)
    # Exactly ONE telegram row — the origin session, not a home duplicate.
    tg_rows = [k for k in keys if k[0] == "telegram"]
    assert tg_rows == [("telegram", "chat-42", "thread-7")]
    # Discord has no origin here; the home fallback may cover it, but the
    # critical invariant is: one subscription per (platform, chat, thread).
    assert len(keys) == len(set(keys))
    assert all(
        s["delivery_mode"] == "notify+wake" for s in subs if s["chat_id"] == "chat-42"
    )


def test_tui_origin_create_not_duplicated_by_home_fallback(
    monkeypatch,
    worker_env,
    with_home_channels_env,
):
    """TUI-origin creates keep their tui subscription untouched by the
    home fallback (requirement: preserve origin subscription)."""
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "tui-session-abc")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "tui create stays tui",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True

    subs = _list_subs_for_task(d["task_id"])
    tui_rows = [s for s in subs if s["platform"] == "tui"]
    assert len(tui_rows) == 1
    assert tui_rows[0]["chat_id"] == "tui-session-abc"


def test_subscribe_to_homes_is_idempotent_no_duplicates(
    monkeypatch,
    worker_env,
    with_home_channels_env,
):
    """Calling the home-fallback twice for the same task writes no extra
    rows (INSERT OR IGNORE on the subscription primary key)."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="idem potens", assignee="w1")
        first = kb.subscribe_task_to_configured_homes(conn, tid)
        count_after_first = len(kb.list_notify_subs(conn, tid))
        second = kb.subscribe_task_to_configured_homes(conn, tid)
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert first is True and second is True
    assert count_after_first == len(subs) == 2
    assert _sub_keys(subs) == [
        ("discord", "9999999", ""),
        ("telegram", "1234567", "42"),
    ]


def test_discord_home_row_uses_channel_chat_type(monkeypatch, worker_env, with_home_channels_env):
    """Security finding: Discord homes are guild text CHANNELS — the row must
    not be stamped chat_type='dm' (wrong wake/session source shape)."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="discord channel type", assignee="w1")
        kb.subscribe_task_to_configured_homes(conn, tid)
        subs = {s["platform"]: s for s in kb.list_notify_subs(conn, tid)}
    finally:
        conn.close()

    assert subs["discord"]["chat_type"] == "channel"
    assert subs["telegram"]["chat_type"] == "dm"


def test_home_failure_does_not_break_task_creation(
    monkeypatch,
    worker_env,
    with_home_channels_env,
):
    """If the homes config explodes mid-create, the task still persists and
    the create response stays ok=True (resilience requirement)."""
    from tools import kanban_tools as kt

    _clear_session_env(monkeypatch)

    from tools import kanban_tools as _mod_self

    def _broken_homes(conn, task_id):
        raise RuntimeError("homes config exploded")

    monkeypatch.setattr(_mod_self, "_subscribe_task_to_configured_homes", _broken_homes)

    out = kt._handle_create({
        "title": "survives broken homes",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["task_id"]
