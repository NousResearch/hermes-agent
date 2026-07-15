"""Authorization boundary between Kanban lifecycle workers and routers."""
from __future__ import annotations

import json

import pytest


LIFECYCLE_TOOLS = {
    "kanban_show",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
}
ROUTING_TOOLS = {
    "kanban_list",
    "kanban_create",
    "kanban_link",
    "kanban_unblock",
}


def _kanban_schema_names():
    import tools.kanban_tools  # noqa: F401 - ensure registration
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    return {
        item["function"]["name"]
        for item in schema
        if item.get("function", {}).get("name", "").startswith("kanban_")
    }


def test_routing_profiles_default_is_fail_closed():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["routing_profiles"] == []


def test_dispatcher_leaf_schema_is_lifecycle_only(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_leaf")
    monkeypatch.delenv("HERMES_KANBAN_ROUTING_AUTHORITY", raising=False)

    names = _kanban_schema_names()

    assert names == LIFECYCLE_TOOLS
    assert names.isdisjoint(ROUTING_TOOLS)


def test_dispatcher_router_schema_includes_routing_tools(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_router")
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_AUTHORITY", "1")

    names = _kanban_schema_names()

    assert names == LIFECYCLE_TOOLS | ROUTING_TOOLS


@pytest.fixture
def leaf_board(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "leaf")
    monkeypatch.delenv("HERMES_KANBAN_ROUTING_AUTHORITY", raising=False)
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        current = kb.create_task(conn, title="leaf", assignee="leaf")
        other = kb.create_task(conn, title="other", assignee="peer")
        kb.claim_task(conn, current)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", current)
    return current, other


def _assert_routing_denied(raw: str):
    payload = json.loads(raw)
    assert "error" in payload
    assert "routing authority" in payload["error"].lower()


def test_leaf_runtime_rejects_task_creation(leaf_board):
    from tools import kanban_tools as kt

    _assert_routing_denied(
        kt._handle_create({"title": "escape", "assignee": "peer"})
    )


def test_leaf_runtime_rejects_cross_board_access(leaf_board):
    from tools import kanban_tools as kt

    current, _ = leaf_board
    payload = json.loads(kt._handle_show({"task_id": current, "board": "alt"}))
    assert "pinned to board" in payload.get("error", "").lower()


def test_routing_worker_is_still_pinned_to_dispatch_board(monkeypatch, leaf_board):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_ROUTING_AUTHORITY", "1")
    payload = json.loads(kt._handle_list({"board": "alt"}))
    assert "pinned to board" in payload.get("error", "").lower()


def test_leaf_runtime_rejects_foreign_show(leaf_board):
    from tools import kanban_tools as kt

    _, other = leaf_board
    _assert_routing_denied(kt._handle_show({"task_id": other}))


def test_leaf_can_comment_on_own_task(leaf_board):
    from tools import kanban_tools as kt

    current, _ = leaf_board
    payload = json.loads(
        kt._handle_comment({"task_id": current, "body": "bounded progress note"})
    )
    assert payload.get("ok") is True


def test_leaf_runtime_rejects_foreign_comment(leaf_board):
    from tools import kanban_tools as kt

    _, other = leaf_board
    _assert_routing_denied(
        kt._handle_comment({"task_id": other, "body": "cross-task handoff"})
    )


def test_leaf_runtime_rejects_dependency_link(leaf_board):
    from tools import kanban_tools as kt

    current, other = leaf_board
    _assert_routing_denied(
        kt._handle_link({"parent_id": current, "child_id": other})
    )
