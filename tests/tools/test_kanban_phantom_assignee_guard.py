"""Behavioural tests for the kanban phantom-assignee guard.

A card created with a copyable placeholder token — the historic ``reviewer`` /
``writer`` / ``researcher-a`` examples the ``kanban_create`` schema used to
advertise — is never dispatched: the dispatcher only spawns a profile it can
resolve, so the card sits in ``ready`` forever and the work is silently
stranded (#106163).

The guard is deliberately NARROWER than the reviewer-side check that already
ships: it rejects only the documented copyable tokens (when they are not real
profiles), angle-bracket placeholders, and whitespace-only names. An arbitrary
not-yet-installed assignee stays accepted, because upstream supports orchestrator
fan-out to seats that are registered outside ``profiles/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_home(monkeypatch, tmp_path):
    """Isolated HERMES_HOME/HOME so profile lookup sees only fixture profiles.

    ``get_profile_dir`` is HOME-anchored (``~/.hermes/profiles``), so both the env
    var and ``Path.home`` have to move or the guard would read the developer's
    real roster.
    """
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return home


def _make_profile(home: Path, name: str) -> Path:
    """A profile dir only counts as live when it carries an identity marker."""
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    return profile_dir


def _board_task_count(home: Path) -> int:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        return len(kb.list_tasks(conn))
    finally:
        conn.close()


@pytest.mark.parametrize("placeholders", [
    "reviewer",        # the tokens the schema advertised as examples
    "writer",
    "researcher-a",
    "<profile>",       # angle-bracket placeholder
    "<your-profile>",
    "   ",             # whitespace-only
])
def test_create_rejects_placeholder_assignee_without_creating_a_card(temp_home, placeholders):
    """A placeholder assignee must fail closed BEFORE the board is mutated."""
    from tools import kanban_tools as kt

    before = _board_task_count(temp_home)
    out = kt._handle_create({"title": "stranded work", "assignee": placeholders})
    payload = json.loads(out)

    assert "error" in payload, f"{placeholders!r} was accepted as an assignee"
    assert payload.get("ok") is not True
    assert _board_task_count(temp_home) == before, "the rejected create still wrote a card"


def test_create_accepts_real_profiles_and_uninstalled_seats(temp_home):
    """Positive controls: a real profile is accepted, and so is a seat that is not
    installed yet (upstream supports registered/external seats — the guard must not
    invent a second seat registry or turn into a global assignee allowlist)."""
    from tools import kanban_tools as kt

    _make_profile(temp_home, "reviewer")  # a real profile may be named like a token
    _make_profile(temp_home, "researcher-a")

    for name in ("reviewer", "researcher-a", "peer"):
        payload = json.loads(kt._handle_create({"title": f"for {name}", "assignee": name}))
        assert payload.get("ok") is True, f"{name!r} should be accepted: {payload}"


def test_create_schema_does_not_advertise_copyable_placeholder_examples():
    """The model copies what the schema shows: the assignee description must not
    list fake profile names as examples."""
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA

    description = KANBAN_CREATE_SCHEMA["parameters"]["properties"]["assignee"]["description"]
    for token in ("researcher-a", "reviewer", "writer"):
        assert token not in description, (
            f"{token!r} is still advertised as an example assignee in the schema")