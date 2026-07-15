"""Tests for /plan handling in tui_gateway.

The TUI previously routed ``/plan`` through the skill scan (advisory only), so
TUI sessions never entered the persisted, code-enforced plan state the CLI and
messaging gateway use. ``command.dispatch`` now drives the SAME PlanManager
state; the dispatch guard in ``agent/tool_executor`` reads that state from the
shared SessionDB, so enforcement holds for TUI turns.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Seed a minimal `plan` skill so `/plan <request>` resolves it by name.
    skill_dir = home / "skills" / "plan"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: plan\n"
        "description: Write a markdown implementation plan instead of executing.\n"
        "---\n"
        "# Plan skill\n\nWrite a plan under `.hermes/plans/`; do not execute.\n"
    )

    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-plan-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


def test_bare_plan_enters_and_persists_enforced_state(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="plan", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "Plan mode on" in r["result"]["output"]

    from hermes_cli.plan_mode import PlanManager

    assert PlanManager(session_key).is_active() is True


def test_bare_plan_enables_dispatch_guard(server, session):
    """The crux: entering plan mode in the TUI blocks mutating tools via the
    shared dispatch guard (keyed on the same session id)."""
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="plan", arg="", session_id=sid)

    from hermes_cli.plan_mode import tool_block_reason

    # Mutating tool blocked; read-only tool allowed.
    assert tool_block_reason(session_key, "terminal", {"command": "ls"}) is not None
    assert tool_block_reason(session_key, "read_file", {"path": "a.py"}) is None


def test_plan_status_reports_state(server, session):
    sid, _, _ = session
    _call(server, "command.dispatch", name="plan", arg="", session_id=sid)
    r = _call(server, "command.dispatch", name="plan", arg="status", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "planning" in r["result"]["output"].lower()


def test_plan_approve_unlocks(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="plan", arg="", session_id=sid)
    r = _call(server, "command.dispatch", name="plan", arg="approve", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "approved" in r["result"]["output"].lower()

    from hermes_cli.plan_mode import PlanManager, tool_block_reason

    assert PlanManager(session_key).is_active() is False
    # Approved → mutations unlocked.
    assert tool_block_reason(session_key, "terminal", {"command": "ls"}) is None


def test_plan_exit_discards_and_unlocks(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="plan", arg="", session_id=sid)
    r = _call(server, "command.dispatch", name="plan", arg="exit", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "discarded" in r["result"]["output"].lower()

    from hermes_cli.plan_mode import PlanManager, tool_block_reason

    assert PlanManager(session_key).is_active() is False
    assert tool_block_reason(session_key, "terminal", {"command": "ls"}) is None


def test_plan_reject_keeps_planning(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="plan", arg="", session_id=sid)
    r = _call(server, "command.dispatch", name="plan", arg="reject too vague", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "too vague" in r["result"]["output"]

    from hermes_cli.plan_mode import PlanManager

    assert PlanManager(session_key).is_active() is True


def test_plan_request_enters_mode_and_seeds_skill(server, session):
    sid, session_key, _ = session
    r = _call(
        server,
        "command.dispatch",
        name="plan",
        arg="implement the login flow",
        session_id=sid,
    )
    result = r["result"]
    # Enters enforcement mode …
    from hermes_cli.plan_mode import PlanManager

    assert PlanManager(session_key).is_active() is True
    # … AND seeds the documented plan skill with the request (args preserved).
    assert result["type"] == "send"
    assert "Plan mode on" in result["notice"]
    assert "implement the login flow" in result["message"]
    assert "plan" in result["message"].lower()
