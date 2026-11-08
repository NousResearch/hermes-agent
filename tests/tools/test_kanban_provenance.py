"""Worker provenance is not a dependency edge or a transient runtime session."""
import json

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("linked,explicit", [(False, None), (True, None), (False, "override")])
def test_worker_create_keeps_durable_origin(tmp_path, monkeypatch, linked, explicit):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
    from tools import kanban_tools as kt, async_delegation
    from gateway.session_context import set_session_vars, clear_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        owner = kb.create_task(
            conn, title="owner", session_id="durable", assignee="default",
        )
        kn.add_notify_sub(conn, task_id=owner, platform="discord", chat_id="chat",
                          user_id="user", notifier_profile="default", delivery_mode="notify",
                          delivery_metadata={"scope_id": "guild", "parent_chat_id": "forum"})
        kb.claim_task(conn, owner)
        expected = kn.list_notify_subs(conn, owner)[0]
        run_id = kb.get_task(conn, owner).current_run_id
    if explicit:
        state = SessionDB(db_path=tmp_path / "state.db")
        state.create_session(explicit, source="cli")
        state.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", owner)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
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


@pytest.mark.parametrize(("session_id", "persisted"), [
    ("phantom-session", False),
    ("persisted-session", True),
])
def test_tool_create_only_stamps_persisted_ambient_session(tmp_path, monkeypatch, session_id, persisted):
    """Ambient worker ids are provenance only after their state.db row exists."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt
    from gateway.session_context import scoped_current_session_id

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    state = SessionDB(db_path=tmp_path / "state.db")
    if persisted:
        state.create_session(session_id, source="cli")
    state.close()

    # Bound the way agent construction publishes it (ContextVar); a bare os.environ value is
    # masked once a surface has cleared its session vars, so it is not a stand-in here. The env
    # var is set too so the reporter's unverified-env path (the pre-fix stamping seam) is exercised.
    monkeypatch.setenv("HERMES_SESSION_ID", session_id)
    with scoped_current_session_id(session_id):
        result = json.loads(kt._handle_create({"title": "child", "assignee": "default"}))
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task.session_id == (session_id if persisted else None)


def test_tool_create_stamps_request_scoped_session_over_process_env(tmp_path, monkeypatch):
    """In a multi-session process os.environ holds the LAST agent built; the request-scoped
    binding names the conversation that actually ordered the card."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt
    from gateway.session_context import scoped_current_session_id

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", "other-session")
    kb.init_db()
    state = SessionDB(db_path=tmp_path / "state.db")
    for sid in ("other-session", "ordering-session"):
        state.create_session(sid, source="cli")
    state.close()

    with scoped_current_session_id("ordering-session"):
        result = json.loads(kt._handle_create({"title": "child", "assignee": "default"}))
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, result["task_id"]).session_id == "ordering-session"


def test_worker_link_rejects_persisted_foreign_nonrunning_child_and_normalizes_ids(tmp_path, monkeypatch):
    """Only the persisted current worker task may be a link child, irrespective of run status."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        own = kb.create_task(conn, title="own", assignee="test-worker")
        kb.claim_task(conn, own)
        run_id = kb.get_task(conn, own).current_run_id
        parent = kb.create_task(conn, title="parent")
        foreign = kb.create_task(conn, title="foreign", assignee="test-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    # String-normalize numeric-like argument payloads before they reach the guard/database.
    own_result = json.loads(kt._handle_link({"parent_id": parent, "child_id": f" {own} "}))
    assert own_result["ok"] is True
    rejected = json.loads(kt._handle_link({"parent_id": parent, "child_id": foreign}))
    assert "foreign child" in rejected["error"]
    with kbc.connect_closing() as conn:
        assert kb.parent_ids(conn, own) == [parent]
        assert kb.parent_ids(conn, foreign) == []
