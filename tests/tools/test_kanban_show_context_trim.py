"""``kanban_show(context=...)``: the optional trimmed worker handoff (#95561).

A card routed through many workers makes every downstream worker re-read the
full parent handoff chain, its own prior attempts and the assignee's recent-work
history. The trim has to shrink what that worker actually pays for, so these
tests pin both halves of the contract:

  * ``build_worker_context(..., mode=...)`` renders the history sections for
    ``full``, drops prior attempts + recent work for ``compact``, and keeps only
    the header + parent handoffs for ``minimal`` — with the default unchanged;
  * ``kanban_show`` exposes the mode, drops the structured logs the trimmed
    handoff already summarises, and errors on an unknown mode instead of
    silently returning the full payload.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def card_with_history(monkeypatch, tmp_path):
    """A claimed card whose full handoff carries every section the trim targets.

    Parent task: completed by the same profile (feeds "recent work by ...").
    Card under test: body + one closed prior attempt + one comment.
    """
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
        parent = kb.create_task(
            conn, title="upstream card", assignee="test-worker",
            body="upstream body")
        assert kb.claim_task(conn, parent) is not None
        assert kb.complete_task(conn, parent, result="upstream shipped")

        tid = kb.create_task(
            conn, title="worker card", assignee="test-worker",
            body="do the thing\n" + "detail line\n" * 40, parents=[parent])
        assert kb.claim_task(conn, tid) is not None
        assert kb.block_task(conn, tid, reason="waiting on a teammate")
        kb.add_comment(conn, tid, "test-worker", "note for the next worker")
    finally:
        conn.close()
    return tid


def test_build_worker_context_modes_trim_history(card_with_history):
    """``compact``/``minimal`` drop the named sections; ``full`` is untouched."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = card_with_history
    conn = kbc.connect()
    try:
        full = kb.build_worker_context(conn, tid)
        compact = kb.build_worker_context(conn, tid, mode="compact")
        minimal = kb.build_worker_context(conn, tid, mode="minimal")

        # The default is what every dispatcher-spawned worker already gets.
        assert full == kb.build_worker_context(conn, tid, mode="full")

        assert "## Body" in full
        assert "## Prior attempts on this task" in full
        assert "## Recent work by @test-worker" in full
        assert "## Parent task results" in full
        assert "## Comment thread" in full

        # compact: the body + parent handoff, no attempt history, no role history.
        assert "## Body" in compact
        assert "## Parent task results" in compact
        assert "## Comment thread" in compact
        assert "## Prior attempts on this task" not in compact
        assert "## Recent work by @test-worker" not in compact

        # minimal: the header spine + parent handoff summaries only.
        assert f"# Kanban task {tid}: worker card" in minimal
        assert "## Parent task results" in minimal
        assert "upstream shipped" in minimal
        assert "## Body" not in minimal
        assert "## Prior attempts on this task" not in minimal
        assert "## Recent work by @test-worker" not in minimal
        assert "## Comment thread" not in minimal

        # The trim is a real reduction, and the rendering says how to escalate.
        assert len(minimal) < len(compact) < len(full)
        assert 'context="full"' in minimal

        # An unknown mode must not fall back to the full handoff.
        with pytest.raises(ValueError):
            kb.build_worker_context(conn, tid, mode="everything")
    finally:
        conn.close()


def test_kanban_show_drops_duplicated_logs_in_trimmed_modes(card_with_history, monkeypatch):
    """The tool payload shrinks too, and the schema enum matches the builder."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    from tools.kanban_tools_schemas import KANBAN_SHOW_SCHEMA

    tid = card_with_history
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    full = json.loads(kt._handle_show({}))
    compact = json.loads(kt._handle_show({"context": "compact"}))
    minimal = json.loads(kt._handle_show({"context": "minimal"}))

    assert {"comments", "events", "runs", "worker_context"} <= set(full)
    assert "context_mode" not in full

    assert compact["context_mode"] == "compact"
    assert compact["task"]["id"] == tid
    assert "comments" in compact
    assert "runs" not in compact
    assert "events" not in compact
    assert "## Prior attempts on this task" not in compact["worker_context"]
    assert "## Recent work by @test-worker" not in compact["worker_context"]

    assert minimal["context_mode"] == "minimal"
    assert minimal["task"]["id"] == tid
    assert "body" not in minimal["task"]
    assert {"comments", "runs", "events"}.isdisjoint(minimal)
    assert "## Body" not in minimal["worker_context"]

    assert len(json.dumps(minimal)) < len(json.dumps(compact)) < len(json.dumps(full))

    # The schema is the only thing the model sees; it must bless the builder's modes.
    assert KANBAN_SHOW_SCHEMA["parameters"]["properties"]["context"]["enum"] == list(kb.KANBAN_CONTEXT_MODES)

    # A typo'd mode is a tool error, never a silent full payload.
    err = kt._handle_show({"context": "medium"})
    assert "context must be one of" in err
