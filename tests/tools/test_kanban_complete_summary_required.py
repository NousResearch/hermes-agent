"""kanban_complete summary contract (reliability gate, observed attended worker runs).

Evidence from that session (messages 3262–3299): GLM-5.3 claimed 19 times to be sending
`summary` while emitting only `{artifacts, board}`; the old "summary OR result" tolerance
turned every replay into a soft rejection the model could ignore forever. Contracts here:

  - the model-facing schema declares `summary` as required and no longer advertises `result`;
  - the exact faulty argument form from message 3262 is rejected, and the card is NOT done;
  - a legacy `result` arg is still accepted alongside `summary` (handler/CLI compatibility)
    but never substitutes for a missing summary;
  - a normal summary completion keeps working unchanged.
"""
from __future__ import annotations

import json

import pytest


# Local copy of the worker fixture pattern (tests/tools/test_kanban_tools.py::worker_env).
@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="summary-contract", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


# --- The exact faulty argument form from message 3262 -----------------------

# The form GLM-5.3 actually emitted 19 times: artifacts + board, no summary, no result.
FAULTY_ARGS_FROM_MESSAGE_3262 = {
    "artifacts": ["/tmp/hermes-test.XXXXXX/report.json"],
    "board": "default",
}


def test_message_3262_arg_form_is_rejected_and_card_not_done(worker_env):
    """{artifacts, board} without summary is a structured rejection; the card stays
    running and no run is completed — nothing may be auto-filled into summary."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = json.loads(kt._handle_complete(dict(FAULTY_ARGS_FROM_MESSAGE_3262)))

    assert out.get("error"), out
    assert "summary is required" in out["error"]
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert task.status == "running", "no card may be done by a summary-less call"
        run = kb.latest_run(conn, worker_env)
        assert run is None or run.outcome != "completed"


def test_repeated_faulty_calls_never_complete_the_card(worker_env):
    """The full observed failure loop — 19 identical faulty calls — must never mark
    the card done nor write a summary/result the model never sent."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    for _ in range(19):
        out = json.loads(kt._handle_complete(dict(FAULTY_ARGS_FROM_MESSAGE_3262)))
        assert out.get("error")

    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert task.status == "running"
        assert not (task.result or ""), task.result


def test_result_only_tool_call_is_rejected(worker_env):
    """A legacy `result` arg never substitutes for a missing summary."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = json.loads(kt._handle_complete({"result": "legacy log line"}))
    assert out.get("error")
    assert "summary is required" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"


def test_whitespace_summary_is_rejected(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = json.loads(kt._handle_complete({"summary": "   \n\t "}))
    assert out.get("error")
    assert "summary is required" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"


# --- Success paths -----------------------------------------------------------

def test_summary_completion_still_works(worker_env):
    """Normal case: a summary-bearing call completes the card unchanged."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = json.loads(kt._handle_complete({"summary": "shipped the fix, tests green"}))
    assert out.get("ok") is True, out
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        run = kb.latest_run(conn, worker_env)
    assert task.status == "done"
    assert run.summary == "shipped the fix, tests green"


def test_legacy_result_alongside_summary_is_stored(worker_env):
    """Legacy compatibility: `result` is still honoured (stored as task.result)
    when a real summary is present — it is removed from the model schema, not
    from the handler."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    out = json.loads(kt._handle_complete({
        "summary": "done with the migration",
        "result": "legacy log line",
    }))
    assert out.get("ok") is True, out
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        run = kb.latest_run(conn, worker_env)
    assert task.status == "done"
    assert task.result == "legacy log line"
    assert run.summary == "done with the migration"


# --- Schema contract ---------------------------------------------------------

def test_schema_requires_summary_and_hides_result():
    """The model-facing schema must make `summary` required and must not advertise
    the legacy `result` parameter."""
    from tools.kanban_tools_schemas import KANBAN_COMPLETE_SCHEMA

    params = KANBAN_COMPLETE_SCHEMA["parameters"]
    assert "summary" in params["required"]
    assert "result" not in params["properties"], (
        "the legacy result field must not be advertised to the model"
    )
    assert "summary" in params["properties"]
    # Every other declared parameter stays optional.
    assert params["required"] == ["summary"]


def test_registered_schema_matches_contract():
    """The registry serves the same contract the schema module declares."""
    from tools.kanban_tools_schemas import KANBAN_COMPLETE_SCHEMA
    from tools.registry import registry

    served = registry.get_schema("kanban_complete")
    assert served["parameters"]["required"] == ["summary"]
    assert "result" not in served["parameters"]["properties"]
    assert served == KANBAN_COMPLETE_SCHEMA


def test_handler_still_accepts_result_as_undeclared_legacy_arg(worker_env):
    """The unknown-parameter gate must not reject a legacy caller passing `result`
    (it is declared in _UNDECLARED_ARGS); the summary requirement is what stops
    result-only calls, not the arg gate."""
    from tools import kanban_tools as kt

    # `result` alone passes the arg gate but fails the summary requirement —
    # the error is the summary contract, not "unknown parameter".
    out = json.loads(kt._handle_complete({"result": "x"}))
    assert "unknown parameter" not in out.get("error", "")
    assert "summary is required" in out.get("error", "")
