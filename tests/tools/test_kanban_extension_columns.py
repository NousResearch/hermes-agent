"""Extension columns for ``kanban_create`` (issue #109800).

A deployment can extend the ``tasks`` table with its own columns (e.g.
``tasks."Token消耗"``, created idempotently by a local migration tool) and
declare them — plus which are mandatory — in the overlay file
``<kanban_home>/kanban/extension-columns.json``.

Contract under test:
  * ``kanban_create``'s tool schema exposes the declared columns (required
    ones in ``required``) so the model can supply them without guessing.
  * Supplied values land in the INSERT (the card is no longer silently
    column-less).
  * A call that omits a deployment-required column is rejected *before* any
    card is created; a value outside a declared enum is rejected too.
  * A missing/malformed overlay, or one that tries to shadow a native field,
    never breaks card creation.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def board_home(monkeypatch, tmp_path):
    """Isolated HERMES_HOME + initialised default kanban board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    for var in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
        "HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _overlay_path(home):
    return home / "kanban" / "extension-columns.json"


def _write_overlay(home, columns):
    path = _overlay_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"columns": columns}, ensure_ascii=False), encoding="utf-8")
    return path


def _add_extension_column(name="Token消耗"):
    """The deployment's idempotent local migration (outside Hermes)."""
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        conn.execute(f'ALTER TABLE tasks ADD COLUMN "{name}" TEXT')


def _read_extension_column(task_id, name="Token消耗"):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        row = conn.execute(f'SELECT "{name}" FROM tasks WHERE id = ?', (task_id,)).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Store level: extra_fields pass-through + validation
# ---------------------------------------------------------------------------

def test_create_task_writes_extension_values_into_insert(board_home):
    """Red before #109800: create_task had no way to populate a deployment column."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _add_extension_column()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="declares token use", assignee="peer",
            extra_fields={"Token消耗": "是"})
    assert _read_extension_column(tid) == "是"


def test_create_task_rejects_extension_column_missing_from_table(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="does not exist on the tasks table"):
            kb.create_task(conn, title="x", assignee="peer",
                           extra_fields={"NotAColumn": 1})


def test_create_task_rejects_managed_column_as_extension(board_home):
    """``extra_fields`` must not double-assign a column create_task owns."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="managed by create_task"):
            kb.create_task(conn, title="x", assignee="peer",
                           extra_fields={"status": "done"})


def test_create_task_rejects_non_scalar_extension_value(board_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    _add_extension_column()
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="must be a scalar"):
            kb.create_task(conn, title="x", assignee="peer",
                           extra_fields={"Token消耗": {"nested": True}})


# ---------------------------------------------------------------------------
# Tool level: schema exposure + required enforcement + write-through
# ---------------------------------------------------------------------------

ENUM_COLUMN = {
    "name": "Token消耗",
    "type": "string",
    "required": True,
    "description": "Whether this card consumes cloud tokens.",
    "enum": ["是", "否", "未知"],
}


def test_schema_exposes_declared_extension_columns(board_home):
    from tools.kanban_tools import _create_schema_overrides

    # Undeclared: the static schema stays untouched.
    assert _create_schema_overrides() == {}

    _write_overlay(board_home, [ENUM_COLUMN])
    overrides = _create_schema_overrides()
    params = overrides["parameters"]
    assert params["properties"]["Token消耗"]["type"] == "string"
    assert params["properties"]["Token消耗"]["enum"] == ["是", "否", "未知"]
    assert "Token消耗" in params["required"]
    # Native fields survive the overlay.
    assert "title" in params["properties"] and "title" in params["required"]

    # Wired into the registry entry, not just importable.
    from tools.registry import registry
    entry = registry.get_entry("kanban_create")
    assert entry is not None and entry.dynamic_schema_overrides is not None
    assert "Token消耗" in entry.dynamic_schema_overrides()["parameters"]["properties"]


def test_tool_creates_card_with_extension_value(board_home):
    from tools import kanban_tools as kt

    _write_overlay(board_home, [ENUM_COLUMN])
    _add_extension_column()
    out = json.loads(kt._handle_create(
        {"title": "child task", "assignee": "peer", "Token消耗": "否"}))
    assert out["ok"] is True, out
    assert _read_extension_column(out["task_id"]) == "否"


def test_tool_rejects_missing_required_extension_column(board_home):
    """The board rule must be enforceable through the native tool: no card is
    created when a deployment-required column is omitted."""
    from tools import kanban_tools as kt

    _write_overlay(board_home, [ENUM_COLUMN])
    _add_extension_column()
    out = json.loads(kt._handle_create({"title": "leaky card", "assignee": "peer"}))
    assert "error" in out, out
    assert "Token消耗" in out["error"]

    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert count == 0


def test_tool_rejects_value_outside_declared_enum(board_home):
    from tools import kanban_tools as kt

    _write_overlay(board_home, [ENUM_COLUMN])
    _add_extension_column()
    out = json.loads(kt._handle_create(
        {"title": "child", "assignee": "peer", "Token消耗": "maybe"}))
    assert "error" in out and "Token消耗" in out["error"]


def test_tool_optional_extension_column_may_be_omitted(board_home):
    from tools import kanban_tools as kt

    _write_overlay(board_home, [dict(ENUM_COLUMN, required=False)])
    _add_extension_column()
    out = json.loads(kt._handle_create({"title": "child", "assignee": "peer"}))
    assert out["ok"] is True, out
    assert _read_extension_column(out["task_id"]) is None


def test_tool_surfaces_missing_table_column_as_clear_error(board_home):
    """Declared but the migration has not run: a clear tool error, not a raw
    sqlite exception."""
    from tools import kanban_tools as kt

    _write_overlay(board_home, [ENUM_COLUMN])
    out = json.loads(kt._handle_create(
        {"title": "child", "assignee": "peer", "Token消耗": "是"}))
    assert "error" in out, out
    assert "does not exist on the tasks table" in out["error"]


def test_overlay_cannot_shadow_native_create_fields(board_home):
    from tools import kanban_tools as kt
    from tools.kanban_tools import _declared_extension_columns, _create_schema_overrides

    _write_overlay(board_home, [{"name": "title", "type": "string", "required": True}])
    assert _declared_extension_columns() == []
    assert _create_schema_overrides() == {}
    out = json.loads(kt._handle_create({"title": "still works", "assignee": "peer"}))
    assert out["ok"] is True, out


# ---------------------------------------------------------------------------
# Overlay loading: fail-open on hand-edited files
# ---------------------------------------------------------------------------

def test_missing_overlay_is_no_extension_columns(board_home):
    from hermes_cli.kanban_extensions import load_extension_columns

    assert load_extension_columns() == []


def test_malformed_overlay_never_breaks_card_creation(board_home):
    from tools import kanban_tools as kt

    path = _overlay_path(board_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    out = json.loads(kt._handle_create({"title": "child", "assignee": "peer"}))
    assert out["ok"] is True, out


def test_overlay_accepts_bare_list_and_skips_bad_entries(board_home):
    from hermes_cli.kanban_extensions import load_extension_columns

    path = _overlay_path(board_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"name": "Token消耗", "type": "string", "required": True},
        {"type": "string"},                       # no name -> skipped
        "not-an-object",                          # skipped
        {"name": "其他", "type": "not-a-type"},     # bad type -> string fallback
        {"name": "Token消耗", "required": False},  # duplicate -> first wins
    ], ensure_ascii=False), encoding="utf-8")
    columns = load_extension_columns()
    assert [c["name"] for c in columns] == ["Token消耗", "其他"]
    assert columns[0]["required"] is True
    assert columns[1]["type"] == "string"
