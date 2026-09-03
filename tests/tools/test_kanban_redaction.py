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
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
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
    secret = "ghp_" + "A" * 40
    kt._handle_comment({"task_id": worker_env, "body": f"token: {secret}"})
    conn = kb.connect()
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
    # Minimal valid-ish JWT (header.payload.sig)
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_5NjP1-iXkpHgcth826Iw"
    )
    kt._handle_block({"reason": f"Bearer {jwt}"})
    conn = kb.connect()
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
    plain = "hello from the pipeline — no secrets here"
    kt._handle_comment({"task_id": worker_env, "body": plain})
    conn = kb.connect()
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
    secret = "ghp_" + "C" * 40
    kt._handle_comment({"task_id": worker_env, "body": f"token: {secret}"})
    conn = kb.connect()
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
    secret = "sk-" + "D" * 48
    kt._handle_complete({"result": f"finished with key={secret}"})
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
    finally:
        conn.close()
    assert run is not None
    stored = run.summary or run.result if hasattr(run, "result") else run.summary or ""
    assert secret not in (stored or "")


def test_kanban_create_body_scrubbed_api_key(worker_env):
    """sk- key in a kanban_create body must be masked before the DB write.

    _handle_create was the only one of the five kanban write paths that
    persisted free text verbatim (#92354) — every other path already
    called redact_sensitive_text(force=True)."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    secret = "sk-" + "a" * 30
    out = json.loads(kt._handle_create({
        "title": f"rotate {secret} now",
        "assignee": "test-worker",
        "body": f"complete using {secret}",
    }))
    assert out["ok"] is True
    conn = kb.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task is not None
    assert secret not in (task.body or "")
    # Head/tail mask survives for debuggability (same rule as other paths).
    assert "sk-" in (task.body or "")
    # Titles are free text too and masked under the same contract —
    # a key pasted into a title would otherwise survive in board listings
    # and dispatcher logs (review follow-up on #92354).
    assert secret not in (task.title or "")
    assert "sk-" in (task.title or "")


def test_kanban_create_secret_free_body_passthrough(worker_env):
    """A secret-free body must pass through byte-for-byte."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    body = "Investigate the flaky e2e suite and report findings."
    out = json.loads(kt._handle_create({
        "title": "plain task",
        "assignee": "test-worker",
        "body": body,
    }))
    assert out["ok"] is True
    conn = kb.connect()
    try:
        task = kb.get_task(conn, out["task_id"])
    finally:
        conn.close()
    assert task is not None
    assert task.body == body
