"""Worker provenance is not a dependency edge or a transient runtime session."""
import json

import pytest


@pytest.mark.parametrize("linked,explicit", [(False, None), (True, None), (False, "override")])
def test_worker_create_keeps_durable_origin(tmp_path, monkeypatch, linked, explicit):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
    from tools import kanban_tools as kt, async_delegation
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        owner = kb.create_task(conn, title="owner", session_id="durable")
        kn.add_notify_sub(conn, task_id=owner, platform="discord", chat_id="chat",
                          user_id="user", notifier_profile="default", delivery_mode="notify",
                          delivery_metadata={"scope_id": "guild", "parent_chat_id": "forum"})
        expected = kn.list_notify_subs(conn, owner)[0]
    monkeypatch.setenv("HERMES_KANBAN_TASK", owner)
    monkeypatch.setenv("HERMES_SESSION_ID", "ephemeral")
    monkeypatch.setattr(async_delegation, "_current_origin_session_id", lambda: "api-origin")
    # Even a matching current channel must not upgrade an inherited passive policy.
    tokens = set_session_vars(platform="discord", chat_id="chat", profile="default")
    try:
        result = json.loads(kt._handle_create(dict(title="child", assignee="default",
                            parents=[owner] if linked else [], session_id=explicit)))
    finally:
        clear_session_vars(tokens)
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child.session_id == (explicit or "durable")
        subs = kn.list_notify_subs(conn, child.id)
        assert len(subs) == 1
        for key in ("platform", "chat_id", "user_id", "delivery_mode", "delivery_metadata", "notifier_profile"):
            assert subs[0][key] == expected[key]
        assert bool(conn.execute("SELECT 1 FROM task_links WHERE child_id = ?", (child.id,)).fetchone()) == linked


def test_tool_subscription_captures_conversation_anchors(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
    from tools import kanban_tools as kt
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    tokens = set_session_vars(platform="discord", chat_id="thread", chat_type="thread",
                             scope_id="guild", parent_chat_id="forum", profile="default")
    try:
        result = json.loads(kt._handle_create(dict(title="direct", assignee="default")))
    finally:
        clear_session_vars(tokens)
    assert result["ok"], result
    with kbc.connect_closing() as conn:
        metadata = kn.list_notify_subs(conn, result["task_id"])[0]["delivery_metadata"]
        assert metadata["scope_id"] == "guild"
        assert metadata["parent_chat_id"] == "forum"


def test_tool_create_persists_and_surfaces_triggering_message_origin(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "daily")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    tokens = set_session_vars(
        platform="telegram", chat_id="-10042", thread_id="2203",
        message_id="767", profile="daily", session_id="20260907_191153_242af06a",
    )
    try:
        result = json.loads(kt._handle_create(
            {"title": "traceable", "assignee": "daily"},
            session_id="20260907_191153_242af06a",
            tool_origin={"message_row_id": 767, "prompt": "Prepare the launch checklist\nnow"},
        ))
    finally:
        clear_session_vars(tokens)

    assert result["ok"], result
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert task.origin == {
            "profile": "daily",
            "session_id": "20260907_191153_242af06a",
            "message_id": "767",
            "message_row_id": 767,
            "platform": "telegram",
            "chat_id": "-10042",
            "thread_id": "2203",
            "prompt_excerpt": "Prepare the launch checklist now",
            "session_link": "/chat?resume=20260907_191153_242af06a&profile=daily",
        }
        worker_context = kb.build_worker_context(conn, task.id)
        assert "## Origin" in worker_context
        assert "Message: 767 (session row 767)" in worker_context
        assert "](/chat?resume=20260907_191153_242af06a&profile=daily)" in worker_context
        shown = json.loads(kt._handle_show({"task_id": task.id}))
        assert shown["task"]["origin"] == task.origin


def test_creation_without_session_keeps_origin_absent_on_legacy_board(tmp_path, monkeypatch):
    import sqlite3

    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "kanban.db"
    with sqlite3.connect(db_path) as legacy:
        legacy.executescript("""
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
                status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, created_by TEXT,
                created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
                workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT,
                claim_lock TEXT, claim_expires INTEGER
            );
        """)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="plain")
        task = kb.get_task(conn, task_id)
        assert task.origin is None
        assert "## Origin" not in kb.build_worker_context(conn, task_id)


def test_origin_redacts_secret_before_persistence_and_display(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "daily")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    secret = "sk-proj-" + ("a" * 40)
    tokens = set_session_vars(
        platform="telegram", chat_id="-10042", message_id="811",
        profile="daily", session_id="session-with-secret",
    )
    try:
        result = json.loads(kt._handle_create(
            {"title": "safe origin", "assignee": "daily"},
            session_id="session-with-secret",
            tool_origin={
                "message_row_id": 767,
                "prompt": f"Prepare the launch checklist using API key {secret} today",
            },
        ))
    finally:
        clear_session_vars(tokens)

    assert result["ok"], result
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert task is not None
        assert task.origin is not None
        shown = json.loads(kt._handle_show({"task_id": task.id}))
        worker_context = kb.build_worker_context(conn, task.id)

    persisted = json.dumps(task.origin)
    displayed = json.dumps(shown)
    for surface in (persisted, displayed, worker_context):
        assert secret not in surface
        assert "sk-proj-" not in surface
        assert "Prepare the launch checklist" in surface
    assert task.origin["prompt_excerpt"] != (
        f"Prepare the launch checklist using API key {secret} today"
    )
