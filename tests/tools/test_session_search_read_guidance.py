"""READ guidance must agree with SCROLL's live-context guard (#136496)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools import session_search_tool  # noqa: F401 — registers the real handler
from tools.registry import registry


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _fill(db, sid, count=35):
    return [db.append_message(sid, role="user", content=f"message {i}") for i in range(count)]


def _call(db, args, **context):
    return json.loads(registry.dispatch("session_search", args, db=db, **context))


def _assert_read(result, ids):
    assert result["success"] is True
    assert result["mode"] == "read"
    assert result["truncated"] is True
    assert result["message_count"] == len(ids)
    assert [m["id"] for m in result["messages"]] == ids[:20] + ids[-10:]
    assert [m["content"] for m in result["messages"]] == [
        f"message {i}" for i in list(range(20)) + list(range(len(ids) - 10, len(ids)))
    ]


def _assert_restricted_guidance(result):
    hint = result["message"].lower()
    assert "around_message_id" not in hint
    assert "scroll" in hint and "unavailable" in hint
    assert "active context" in hint


@pytest.mark.parametrize("relationship", ["self", "parent", "child", "compressed_ancestor", "branched"])
def test_live_lineage_read_succeeds_without_unusable_scroll_instruction(db, relationship):
    db.create_session("parent", source="cli")
    db.create_session("child", source="cli", parent_session_id="parent")
    target, current = "parent", "parent"
    if relationship in ("parent", "branched"):
        current = "child"
    elif relationship in ("child", "compressed_ancestor"):
        target = "child"
    if relationship == "branched":
        db.end_session("parent", "branched")
    elif relationship == "compressed_ancestor":
        db.end_session("parent", "compression")
        db.create_session("delegate", source="cli", parent_session_id="child")
        current = "delegate"
    ids = _fill(db, target)

    read = _call(db, {"session_id": target}, current_session_id=current)
    _assert_read(read, ids)
    scroll = _call(db, {"session_id": target, "around_message_id": ids[0]}, current_session_id=current)
    assert scroll["success"] is False
    assert "current session lineage" in scroll["error"]
    _assert_restricted_guidance(read)


@pytest.mark.parametrize("relationship", ["historical", "absent", "compression", "session_reset", "new_session"])
def test_read_keeps_usable_scroll_instruction_outside_live_context(db, relationship):
    db.create_session("history", source="cli")
    ids = _fill(db, "history")
    context = {}
    if relationship != "absent":
        parent = None if relationship == "historical" else "history"
        db.end_session("history", "cli_exit" if relationship == "historical" else relationship)
        db.create_session("current", source="cli", parent_session_id=parent)
        context["current_session_id"] = "current"

    read = _call(db, {"session_id": "history"}, **context)
    _assert_read(read, ids)
    assert "around_message_id" in read["message"]
    scroll = _call(db, {"session_id": "history", "around_message_id": read["messages"][19]["id"]}, **context)
    assert scroll["success"] is True
    assert any(m["id"] == ids[20] for m in scroll["messages"])


def test_read_excludes_compacted_rows_but_archived_anchor_still_scrolls(db):
    db.create_session("current", source="cli")
    archived = db.append_message("current", role="user", content="archived detail")
    db.archive_and_compact("current", [])
    ids = _fill(db, "current")
    read = _call(db, {"session_id": "current"}, current_session_id="current")
    _assert_read(read, ids)
    assert archived not in [m["id"] for m in read["messages"]]
    scroll = _call(db, {"session_id": "current", "around_message_id": archived}, current_session_id="current")
    assert scroll["success"] is True
    assert any(m["id"] == archived and m["anchor"] for m in scroll["messages"])
    _assert_restricted_guidance(read)


@pytest.mark.parametrize("count", [0, 1, 30])
def test_untruncated_active_read_has_no_pagination_hint(db, count):
    db.create_session("current", source="cli")
    ids = _fill(db, "current", count)
    read = _call(db, {"session_id": "current"}, current_session_id="current")
    assert read["success"] is True
    assert read["truncated"] is False
    assert read["message_count"] == count
    assert [m["id"] for m in read["messages"]] == ids
    assert "message" not in read


@pytest.mark.parametrize("failure", ["missing", "load_error"])
def test_read_errors_do_not_gain_scroll_guidance(db, monkeypatch, failure):
    if failure == "load_error":
        db.create_session("current", source="cli")

        def fail_load(*args, **kwargs):
            raise RuntimeError("database read failed")

        monkeypatch.setattr(db, "get_messages", fail_load)
    read = _call(db, {"session_id": "current"}, current_session_id="current")
    assert read["success"] is False
    assert "error" in read
    assert "messages" not in read
    assert "message" not in read


@pytest.mark.parametrize("embedded_profile", [False, True])
def test_named_profile_read_keeps_scroll_hint_even_if_caller_id_matches(db, tmp_path, monkeypatch, embedded_profile):
    from hermes_cli import profiles

    home = tmp_path / "home"
    other_home = home / "profiles" / "work"
    other_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "work")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: other_home)
    other = SessionDB(other_home / "state.db")
    try:
        other.create_session("current", source="cli")
        ids = _fill(other, "current")
    finally:
        other.close()
    args = {"session_id": "work/current"} if embedded_profile else {"session_id": "current", "profile": "work"}
    read = _call(db, args, current_session_id="current")
    _assert_read(read, ids)
    assert read["link"] == "@session:work/current"
    assert "around_message_id" in read["message"]
    scroll = _call(db, {**args, "around_message_id": ids[0]}, current_session_id="current")
    assert scroll["success"] is True


def test_inline_caller_passes_current_session_to_read_guidance(db):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    db.create_session("current", source="cli")
    ids = _fill(db, "current")
    agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, session_id="current")
    ctx = InlineToolContext(effective_task_id="task", tool_call_id="call")
    read = json.loads(INLINE_TOOL_EXECUTORS["session_search"](agent, {"session_id": "current"}, ctx))
    _assert_read(read, ids)
    _assert_restricted_guidance(read)
