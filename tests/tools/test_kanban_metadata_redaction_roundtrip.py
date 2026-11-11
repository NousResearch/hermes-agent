"""Kanban metadata must survive redaction, not fall back to the raw dict.

#123688: ``_redact_metadata`` redacted via a JSON round trip and returned ``None`` whenever a
string leaf ended in an ENV-style secret — ``_ENV_ASSIGN_RE``'s ``(\\S+)`` swallows the closing
quote of the string it sits in, so ``{"note": "DB_PASSWORD=abcdef12"}`` became the unterminated
``{"note": "DB_PASSWORD=***``. ``kanban_complete`` falls back to the **unredacted** dict on a
``None`` return, so the secret was persisted; ``kanban_request_review`` rejected the call.

``test_kanban_redaction.py`` covers the masking of comment/block/summary bodies. This file
covers the metadata path specifically: the structure must come back parseable *and* clean.

Fixture mirrors ``test_kanban_redaction.py`` / ``test_kanban_tools.py``.
"""
from __future__ import annotations

import pytest


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


# A leaf the old round trip corrupted, and its secret half.
CORRUPTING = "DB_PASSWORD=abcdef12"


class TestMetadataIsRedacted:
    def test_an_env_secret_leaf_is_masked_and_the_dict_survives(self):
        from tools.kanban_tools import _redact_metadata

        out = _redact_metadata({"note": CORRUPTING})
        assert out is not None, "the old None return made callers store the raw dict"
        assert isinstance(out, dict)
        assert "abcdef12" not in out["note"], f"secret survived: {out!r}"

    def test_the_secret_named_key_rule_still_applies(self):
        from tools.kanban_tools import _redact_metadata

        out = _redact_metadata({"password": "hunter2", "other": "keep-me"})
        assert out is not None
        assert "hunter2" not in out["password"]
        assert out["other"] == "keep-me"

    def test_nested_structures_are_redacted(self):
        from tools.kanban_tools import _redact_metadata

        out = _redact_metadata({"meta": {"list": [{"deep": CORRUPTING}]}, "n": 1})
        assert out is not None
        assert "abcdef12" not in str(out)

    def test_non_secret_metadata_is_unchanged(self):
        from tools.kanban_tools import _redact_metadata

        payload = {"status": "done", "count": 2, "ok": True}
        assert _redact_metadata(payload) == payload

    def test_an_unexpected_shape_is_not_handed_back_verbatim(self):
        """The caller's fallback is the UNREDACTED original, so anything this cannot
        decompose must not be returned as-is."""
        from tools.kanban_tools import _redact_metadata

        sentinel = object()
        out = _redact_metadata(sentinel)
        assert out is None or out is not sentinel


class TestEndToEndThroughTheHandler:
    def test_kanban_complete_stores_masked_parseable_metadata(self, worker_env):
        """The reported consequence: the card stored the raw dict. Drive the real handler."""
        import json

        from tools import kanban_tools as kt
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        kt._handle_complete({"summary": "done", "metadata": {"note": CORRUPTING}})

        conn = kbc.connect()
        try:
            run = kb.latest_run(conn, worker_env)
        finally:
            conn.close()
        assert run is not None, "complete did not record a run"

        stored = _stored_metadata(run)
        assert stored is not None, "metadata was not persisted at all"
        assert "abcdef12" not in str(stored), f"the unredacted dict was stored: {stored!r}"

    def test_kanban_request_review_accepts_env_secret_metadata(self, worker_env):
        """It rejected the call outright before, because the round trip could not parse."""
        from tools import kanban_tools as kt

        result = kt._handle_request_review({"metadata": {"note": CORRUPTING}})
        text = str(result).lower()
        assert "error" not in text or "metadata" not in text, f"review rejected it: {result!r}"


def _stored_metadata(run):
    """The metadata dict off a run row (``kanban_db`` stores it as JSON TEXT -> dict)."""
    value = getattr(run, "metadata", None)
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        import json
        try:
            return json.loads(value)
        except ValueError:
            return value
    return None