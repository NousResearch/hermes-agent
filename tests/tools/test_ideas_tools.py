"""Tests for tools/ideas_tools.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return home


def test_ideas_tools_in_default_hermes_cli_toolset(monkeypatch, fresh_home):
    import tools.ideas_tools  # noqa: F401
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    expected = {
        "ideas_list", "ideas_boards", "ideas_show", "ideas_create", "ideas_update",
        "ideas_delete", "ideas_convert",
    }
    assert expected <= names


def test_ideas_list_all_boards_tool(fresh_home):
    from hermes_cli import ideas_db as db
    from tools import ideas_tools as it

    db.create_idea(title="A", board="default")
    db.create_idea(title="B", board="other-board")
    raw = it._handle_list({"all_boards": True})
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["count"] == 2
    assert set(data["boards"]) == {"default", "other-board"}


def test_ideas_create_tool(fresh_home):
    from tools import ideas_tools as it

    raw = it._handle_create(
        {"title": "Agent idea", "body": "notes", "board": "default"},
    )
    data = json.loads(raw)
    assert data["ok"] is True
    assert data["idea"]["title"] == "Agent idea"
    assert data["idea"]["board"] == "default"
