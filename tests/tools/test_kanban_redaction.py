"""Tests: redact_sensitive_text is applied in kanban tool handlers.

Verifies that secrets embedded in kanban_comment body, kanban_complete
summary/result/metadata, and kanban_block reason are masked before the
values reach the DB.  Uses the same worker_env fixture pattern as
test_kanban_tools.py.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Shared fixture — mirrors test_kanban_tools.py
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with a running task; returns the task id."""
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
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


# ---------------------------------------------------------------------------
# Positive tests — secrets are masked
# ---------------------------------------------------------------------------

def test_kanban_comment_body_scrubbed_github_pat(worker_env):
    """ghp_ PAT in comment body must be masked before DB write."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    secret = "ghp_" + "A" * 40
    kt._handle_comment({"task_id": worker_env, "body": f"token: {secret}"})
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
    finally:
        conn.close()
    assert comments, "expected at least one comment"
    stored = comments[-1].body
    assert secret not in stored
    assert stored  # something was stored


def test_kanban_block_reason_scrubbed_jwt(worker_env):
    """JWT in block reason must be masked before DB write."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    # Minimal valid-ish JWT (header.payload.sig)
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_5NjP1-iXkpHgcth826Iw"
    )
    kt._handle_block({"reason": f"Bearer {jwt}"})
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    # block_task stores reason as run.summary
    assert run is not None
    assert run.outcome == "blocked"
    stored = run.summary or ""
    assert jwt not in stored


# ---------------------------------------------------------------------------
# Negative test — plain text passes through unchanged
# ---------------------------------------------------------------------------

def test_kanban_comment_no_secret_passthrough(worker_env):
    """Plain text without credential patterns must pass through unchanged."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    plain = "hello from the pipeline — no secrets here"
    kt._handle_comment({"task_id": worker_env, "body": plain})
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
    finally:
        conn.close()
    stored = comments[-1].body
    assert stored == plain


# ---------------------------------------------------------------------------
# Negative test — force=True bypasses HERMES_REDACT_SECRETS=false
# ---------------------------------------------------------------------------

def test_scrub_respects_force_flag_regardless_of_config(worker_env, monkeypatch):
    """force=True must fire even when HERMES_REDACT_SECRETS=false is set."""
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    secret = "ghp_" + "C" * 40
    kt._handle_comment({"task_id": worker_env, "body": f"token: {secret}"})
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
    finally:
        conn.close()
    stored = comments[-1].body
    assert secret not in stored


# ---------------------------------------------------------------------------
# Negative test — legacy result field is also scrubbed
# ---------------------------------------------------------------------------

def test_kanban_complete_result_field_scrubbed(worker_env):
    """Legacy result field must be scrubbed just like summary."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    secret = "sk-" + "D" * 48
    kt._handle_complete({"result": f"finished with key={secret}"})
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    assert run is not None
    assert run.outcome == "completed"
    stored = run.summary or ""
    assert secret not in stored


# ---------------------------------------------------------------------------
# Metadata whose string leaf ENDS in an ENV-style secret: the old
# serialize-then-redact path ate the closing quote. kanban_complete swallowed
# the JSONDecodeError and stored the UNREDACTED dict; request_review errored.
# ---------------------------------------------------------------------------

def test_kanban_complete_metadata_env_secret_masked_and_parseable(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    out = json.loads(kt._handle_complete({
        "task_id": worker_env,
        "summary": "done",
        "metadata": {"x": "DB_PASSWORD=abcdef12", "n": 3},
    }))
    assert "error" not in out, out
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    meta = run.metadata
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta["n"] == 3
    assert meta["x"].startswith("DB_PASSWORD=")
    assert "abcdef12" not in json.dumps(meta)


def test_kanban_request_review_metadata_env_secret_masked(worker_env, monkeypatch):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    # Worker ownership: request_review refuses a live claim without it.
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    out = json.loads(kt._handle_request_review({
        "task_id": worker_env,
        "summary": "impl done, tests pass",
        "metadata": {"note": "PASS=174"},
    }))
    assert "error" not in out, out
    assert "174" not in json.dumps(out)
