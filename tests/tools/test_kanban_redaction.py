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
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
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
    stored = run.summary or run.result if hasattr(run, "result") else run.summary or ""
    assert secret not in (stored or "")


def test_kanban_complete_corrupting_metadata_stays_structured(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    secret = "x-api-key: abc123"
    response = json.loads(kt._handle_complete({
        "summary": "done",
        "metadata": {"headers": secret, "safe": "keep"},
    }))
    assert response["ok"] is True
    conn = connect()
    try:
        metadata = kb.latest_run(conn, worker_env).metadata
    finally:
        conn.close()
    assert metadata["safe"] == "keep"
    assert secret not in json.dumps(metadata)


def test_kanban_request_review_corrupting_metadata_stays_structured(worker_env, monkeypatch):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    secret = "x-api-key: abc123"
    conn = connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    assert run is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    response = json.loads(kt._handle_request_review({
        "summary": "ready for review",
        "metadata": {"headers": secret, "safe": "keep"},
    }))
    assert "error" not in response, response
    conn = connect()
    try:
        metadata = kb.latest_run(conn, worker_env).metadata
    finally:
        conn.close()
    assert metadata is not None
    assert metadata["safe"] == "keep"
    assert secret not in json.dumps(metadata)
