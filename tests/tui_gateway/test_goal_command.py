"""Tests for /goal handling in tui_gateway.

The TUI routes ``/goal`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_goal_command`` queues the kickoff message onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/goal`` directly in the server and return a
``{"type": "send", "notice": ..., "message": ...}`` payload the TUI client
uses to render a system line and fire the kickoff prompt.
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

    # Bust the goal-module DB cache so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    # Pin config resolution to the isolated HERMES_HOME. Sibling test
    # files (test_billing_rpc, test_delegation_session_lifecycle,
    # test_gateway_owned_session_reap, ...) import tui_gateway.server at
    # collection time — BEFORE the conftest env isolation runs — so the
    # module-level ``_hermes_home = get_hermes_home()`` snapshot freezes
    # the developer's real home. When any of them precede this file in
    # the same process, ``importlib.import_module`` returns that cached
    # module and ``_load_cfg()`` would read the REAL config.yaml (e.g. a
    # local MoA preset) instead of the one ``_write_moa_config`` writes.
    # Also reset the mtime-keyed config cache; monkeypatch restores the
    # originals on teardown so nothing leaks to later tests either.
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    # Reset module-level session state without re-importing. importlib.reload
    # would re-register the module's atexit hooks (ThreadPoolExecutor
    # shutdown, _shutdown_sessions); the duplicates race the stderr
    # buffer at interpreter shutdown and surface as Fatal Python error:
    # _enter_buffered_busy. Clearing the per-session dicts gives the
    # next test a clean slate.
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-goal-session-1"
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


# ── command.dispatch /goal ────────────────────────────────────────────


def test_goal_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


# ── slash.exec /goal routing ──────────────────────────────────────────


def test_slash_exec_routes_goal_to_command_dispatch(server, session):
    """slash.exec must route /goal directly to command.dispatch internally
    instead of returning an error.  Previously the 4018 error required the
    TUI client to retry via command.dispatch, but some clients failed the
    fallback, leaving the command empty ("empty command")."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="goal status", session_id=sid)
    # Should succeed by routing to command.dispatch internally
    assert "result" in r
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_pending_input_commands_includes_goal(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'goal' — removing it would
    silently re-break the TUI."""
    assert "goal" in server._PENDING_INPUT_COMMANDS


# ── command.dispatch /goal --file <path> ──────────────────────────────


def test_goal_file_reads_tempfile_relative_to_session_cwd(server, session, tmp_path):
    """``--file`` resolves a relative path against the session's cwd (the
    backend host working directory), parses the contract, persists it, and
    returns the resolved goal text in the ``send.message`` kickoff."""
    sid, session_key, s = session
    s["cwd"] = str(tmp_path)  # session workspace on the Hermes backend host
    goal_path = tmp_path / "goal.txt"
    goal_path.write_text("Ship café feature\n\nverify: pytest -q\n", encoding="utf-8")

    r = _call(server, "command.dispatch", name="goal", arg="--file goal.txt", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "Ship café feature"
    assert "Goal set" in result["notice"]
    assert "Completion contract" in result["notice"]

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    assert mgr.state is not None
    assert mgr.state.goal == "Ship café feature"
    assert mgr.state.contract.verification == "pytest -q"
    assert mgr.state.status == "active"


def test_goal_file_quoted_path_preserved_through_slash_exec(server, session, tmp_path):
    """A quoted path with spaces must survive ``slash.exec`` routing into
    ``command.dispatch`` (the Desktop and TUI composer both go through
    slash.exec)."""
    sid, _, s = session
    s["cwd"] = str(tmp_path)
    goal_path = tmp_path / "release goal.txt"
    goal_path.write_text("ship it", encoding="utf-8")

    r = _call(server, "slash.exec", command=f'goal --file "{goal_path}"', session_id=sid)
    assert "result" in r
    assert r["result"]["message"] == "ship it"


def test_goal_file_content_resembling_subcommand_stored_as_data(server, session, tmp_path):
    """A file whose content is ``status`` becomes a goal named ``status``,
    not a status query — subcommands are classified before the file is read."""
    sid, _, s = session
    s["cwd"] = str(tmp_path)
    goal_path = tmp_path / "goal.txt"
    goal_path.write_text("status", encoding="utf-8")

    r = _call(server, "command.dispatch", name="goal", arg="--file goal.txt", session_id=sid)
    assert r["result"]["type"] == "send"
    assert r["result"]["message"] == "status"


def test_goal_file_read_error_preserves_existing_goal(server, session, tmp_path):
    """A failed load is atomic: an already-active goal and the kickoff are
    left untouched — no partial state, no enqueue."""
    sid, session_key, s = session
    s["cwd"] = str(tmp_path)
    # Set an active goal first.
    _call(server, "command.dispatch", name="goal", arg="original goal", session_id=sid)

    missing = "does-not-exist.txt"
    r = _call(server, "command.dispatch", name="goal", arg=f"--file {missing}", session_id=sid)
    assert "error" in r
    assert "file not found" in r["error"]["message"]

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    # The original goal survives the failed --file load.
    assert mgr.state.goal == "original goal"
    assert mgr.state.status == "active"


def test_goal_file_remote_only_session_cwd_rejects_relative(server, session, tmp_path):
    """When the session cwd doesn't exist on the backend host (an SSH/Docker
    terminal backend whose workspace lives only inside the backend), a
    relative path is rejected instead of silently using the process cwd."""
    sid, _, s = session
    s["cwd"] = str(tmp_path / "remote-only-not-on-host")
    r = _call(server, "command.dispatch", name="goal", arg="--file goal.txt", session_id=sid)
    assert "error" in r
    assert "absolute path" in r["error"]["message"]


# ── command.dispatch /moa ────────────────────────────────────────────

def _write_moa_config(home, text):
    cfg_path = home / "config.yaml"
    cfg_path.write_text(text)


def test_moa_bare_returns_usage(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="", session_id=sid)
    # Bare /moa is usage-only now; switching to a preset is via the model picker.
    assert "error" in r
    assert "model_override" not in s


