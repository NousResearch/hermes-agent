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


@pytest.mark.parametrize("trusted", ["host-worker-runtime", None])
def test_cross_board_worker_create_keeps_legacy_session_fallback(
    tmp_path, monkeypatch, trusted
):
    from agent.delegation_context import is_dispatcher_owned_worker_context
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import async_delegation
    import model_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "source")
    monkeypatch.setenv("HERMES_SESSION_ID", "legacy-worker-runtime")
    monkeypatch.setattr(async_delegation, "_current_origin_session_id", lambda: "")
    with kbc.connect_closing(board="source") as conn:
        owner = kb.create_task(conn, title="owner", session_id="durable-owner-session")
    monkeypatch.setenv("HERMES_KANBAN_TASK", owner)

    tokens = set_session_vars(platform="discord", chat_id="thread")
    try:
        assert is_dispatcher_owned_worker_context()
        result = json.loads(
            model_tools.handle_function_call(
                "kanban_create",
                {"title": "cross-board", "assignee": "default", "board": "destination"},
                session_id=trusted,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["ok"], result
    with kbc.connect_closing(board="destination") as conn:
        assert kb.get_task(conn, owner) is None
        child = kb.get_task(conn, result["task_id"])
        assert child is not None
        assert child.session_id == "legacy-worker-runtime"
        assert not conn.execute(
            "SELECT 1 FROM task_links WHERE child_id = ?", (child.id,)
        ).fetchone()


def test_direct_create_prefers_trusted_handler_session_over_polluted_state(
    tmp_path, monkeypatch
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import async_delegation
    import model_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", "polluted-child-session")
    monkeypatch.setattr(async_delegation, "_current_origin_session_id", lambda: "")
    kb.init_db()
    tokens = set_session_vars(
        platform="discord",
        chat_id="thread",
        session_id="internal-child-context",
    )
    try:
        result = json.loads(
            model_tools.handle_function_call(
                "kanban_create",
                {"title": "direct", "assignee": "default"},
                session_id="coordinator-session",
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["ok"], result
    with kbc.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child is not None
        assert child.session_id == "coordinator-session"


@pytest.mark.parametrize(
    "explicit,trusted,api_origin,platform,legacy,expected",
    [
        ("explicit", "trusted", "api", "discord", "legacy", "explicit"),
        (None, "trusted", "api", "discord", "legacy", "trusted"),
        (None, None, "api", "discord", "legacy", "api"),
        (None, None, "", "telegram", "legacy", "legacy"),
        (None, None, "", "discord", "legacy", ""),
    ],
)
def test_non_worker_create_session_precedence(
    tmp_path,
    monkeypatch,
    explicit,
    trusted,
    api_origin,
    platform,
    legacy,
    expected,
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import async_delegation
    import model_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", legacy)
    monkeypatch.setattr(
        async_delegation, "_current_origin_session_id", lambda: api_origin
    )
    kb.init_db()
    tokens = set_session_vars(platform=platform, chat_id="chat")
    args = {"title": "direct", "assignee": "default"}
    if explicit:
        args["session_id"] = explicit
    try:
        result = json.loads(
            model_tools.handle_function_call(
                "kanban_create", args, session_id=trusted
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["ok"], result
    with kbc.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child is not None
        assert child.session_id == expected


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
