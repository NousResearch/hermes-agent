"""``kanban_create`` threads ``reasoning_effort`` to the task row (#125613 review)."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools  # noqa: F401  register handlers

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _create(**extra):
    from tools.registry import registry

    return json.loads(registry.dispatch(
        "kanban_create", {"title": "t", "assignee": "builder", **extra}))


def _effort(kb, tid):
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid).reasoning_effort


@pytest.mark.parametrize("given, stored", [
    ("xhigh", "xhigh"), ("HIGH", "high"), ("none", "none"), ("", None)])
def test_create_stores_reasoning_effort(board, given, stored):
    result = _create(reasoning_effort=given)
    assert result.get("ok"), result
    assert _effort(board, result["task_id"]) == stored


def test_create_without_effort_inherits_profile(board):
    result = _create()
    assert result.get("ok"), result
    assert _effort(board, result["task_id"]) is None


def test_create_rejects_unknown_effort_and_creates_nothing(board):
    from hermes_cli import kanban_db_connect as kbc

    result = _create(reasoning_effort="extreme")
    assert "reasoning_effort" in result.get("error", ""), result
    with kbc.connect_closing() as conn:
        assert not board.list_tasks(conn)


def test_schema_enum_matches_db_validation():
    from hermes_cli.kanban_db import normalize_reasoning_effort
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA

    enum = KANBAN_CREATE_SCHEMA["parameters"]["properties"]["reasoning_effort"]["enum"]
    for value in enum:
        normalize_reasoning_effort(value)  # every advertised value is storable
