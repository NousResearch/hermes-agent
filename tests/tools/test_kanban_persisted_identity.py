"""Persisted board identity resolves the ACTIVE profile, never a generic label.

Regression: ``kanban_comment``/``kanban_create`` persisted ``"worker"`` whenever
the dispatcher had not pinned ``HERMES_PROFILE``, even though the running Hermes
profile was discoverable from ``HERMES_HOME`` via
``hermes_cli.profiles.get_active_profile_name``. Comment authors are injected
into future workers' prompts and ``created_by`` gates completion audits, so a
generic label silently loses handoff provenance.

Contract under test (``tools.kanban_tools._persisted_identity``): environment
profile (``HERMES_PROFILE_NAME``/``HERMES_PROFILE``) → active profile API →
generic ``"worker"`` fallback. Caller args can never override the persisted
identity (#19713).
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def board_env(tmp_path, monkeypatch):
    """Worker on an isolated default board whose ``HERMES_HOME`` sits under a
    NAMED profile dir (``<root>/profiles/qa-bot``) — the shape a profile launch
    has when the dispatcher did not export ``HERMES_PROFILE``. The ``profiles``
    parent makes ``get_default_hermes_root()`` resolve to ``<root>`` on every
    platform, so the board DB stays inside tmp_path."""
    profile_home = tmp_path / "hroot" / "profiles" / "qa-bot"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="identity-test", assignee="qa-bot")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _last_comment_author(tid: str) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.list_comments(conn, tid)[-1].author
    finally:
        conn.close()


def _created_by(tid: str) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.get_task(conn, tid).created_by
    finally:
        conn.close()


def test_env_profile_takes_precedence(board_env, monkeypatch):
    """A dispatcher-pinned ``HERMES_PROFILE`` wins over the HERMES_HOME-derived
    active profile for both persisted record kinds."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_PROFILE", "pinned-bot")

    out = json.loads(kt._handle_comment({"task_id": board_env, "body": "from env"}))
    assert out["ok"]
    assert _last_comment_author(board_env) == "pinned-bot"

    child = json.loads(kt._handle_create(
        {"title": "env child", "assignee": "peer", "parents": [board_env]}))
    assert child["ok"]
    assert _created_by(child["task_id"]) == "pinned-bot"


def test_active_profile_used_when_env_absent(board_env):
    """The regression: no ``HERMES_PROFILE`` in the environment, but
    ``HERMES_HOME`` names the active profile — persisted records must carry
    that profile, not the generic ``"worker"`` label."""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_comment(
        {"task_id": board_env, "body": "from active profile"}))
    assert out["ok"]
    assert _last_comment_author(board_env) == "qa-bot"

    child = json.loads(kt._handle_create(
        {"title": "profile child", "assignee": "peer", "parents": [board_env]}))
    assert child["ok"]
    assert _created_by(child["task_id"]) == "qa-bot"


def _raiser():
    raise RuntimeError("profile store unavailable")


@pytest.mark.parametrize("stub", [lambda: "", _raiser])
def test_generic_fallback_without_any_profile(board_env, monkeypatch, stub):
    """No env profile AND an unusable active-profile API → generic ``"worker"``
    (records stay attributable to *a* worker instead of failing the call)."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", stub)
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_comment({"task_id": board_env, "body": "anon"}))
    assert out["ok"]
    assert _last_comment_author(board_env) == "worker"

    child = json.loads(kt._handle_create(
        {"title": "anon child", "assignee": "peer", "parents": [board_env]}))
    assert child["ok"]
    assert _created_by(child["task_id"]) == "worker"


def _comment_authors(tid: str) -> list[str]:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return [c.author for c in kb.list_comments(conn, tid)]
    finally:
        conn.close()


def test_caller_args_cannot_override_persisted_identity(board_env):
    """#19713 stays fixed — and got stronger upstream: ``author``/``created_by``
    are undeclared parameters of ``kanban_comment``/``kanban_create``, so the
    handler wrapper rejects the call outright ("Nothing changed") before it can
    reach the handler. Identity can never come from tool args (see also
    ``_persisted_identity``, which has no args surface)."""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_comment(
        {"task_id": board_env, "body": "hi", "author": "hermes-system"}))
    assert not out.get("ok")
    assert "unknown parameter" in out["error"]
    assert _comment_authors(board_env) == []  # nothing persisted

    child = json.loads(kt._handle_create(
        {"title": "forged child", "assignee": "peer", "parents": [board_env],
         "created_by": "hermes-system"}))
    assert not child.get("ok")
    assert "unknown parameter" in child["error"]
