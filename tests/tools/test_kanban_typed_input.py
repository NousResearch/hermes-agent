"""Native typed block contract; all state belongs to a disposable board."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REQUEST = {
    "question": "Which collection should receive the sample?",
    "facts": "Both collections accept the measured volume.",
    "recommendation": "Use collection violet.",
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "fixture")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    (home / "config.yaml").write_text("toolsets: [hermes-cli, kanban]\n")
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools  # register native handlers
    from tools.registry import registry

    with kbc.connect() as conn:
        yield kb, conn, registry


def test_typed_block_persists_exact_request_and_reply_stays_on_card(board):
    kb, conn, registry = board
    tid = kb.create_task(conn, title="Choose collection", initial_status="running")
    args = {
        "task_id": tid,
        "reason": "An owner choice is needed.",
        "kind": "needs_input",
        "input_request": REQUEST,
    }
    result = json.loads(registry.dispatch("kanban_block", args))
    assert result.get("ok"), result
    event = kb.list_events(conn, tid)[-1]
    assert event.kind == "blocked"
    assert event.payload.get("input_request") == REQUEST
    schema = registry.get_entry("kanban_block").schema
    import jsonschema

    jsonschema.validate(args, schema["parameters"])
    jsonschema.validate(REQUEST, schema["parameters"]["properties"]["input_request"])
    # Same-card reply path remains available; no new sequencing policy.
    assert "error" in json.loads(
        registry.dispatch("kanban_comment", {"task_id": tid, "body": ""})
    )
    assert kb.get_task(conn, tid).status == "blocked"
    answer = "Use collection violet."
    comment = json.loads(
        registry.dispatch("kanban_comment", {"task_id": tid, "body": answer})
    )
    shown = json.loads(registry.dispatch("kanban_show", {"task_id": tid}))
    assert comment.get("ok") and shown["task"]["id"] == tid
    assert any(c["body"] == answer for c in shown["comments"])
    assert shown["task"]["status"] == "blocked"
    assert json.loads(registry.dispatch("kanban_unblock", {"task_id": tid})).get("ok")
    assert kb.get_task(conn, tid).status == "ready"
    # A later legacy block must not inherit the earlier typed request.
    assert json.loads(
        registry.dispatch(
            "kanban_block",
            {"task_id": tid, "reason": "Old note", "kind": "needs_input"},
        )
    ).get("ok")
    assert "input_request" not in kb.list_events(conn, tid)[-1].payload


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("needs_input", {}),
        ("needs_input", []),
        ("needs_input", "question"),
        ("needs_input", {**REQUEST, "facts": "  "}),
        ("needs_input", {**REQUEST, "question": 123}),
        ("needs_input", {**REQUEST, "recommendation": None}),
        ("needs_input", {**REQUEST, "extra": "ambiguous"}),
        ("capability", REQUEST),
        (None, REQUEST),
    ],
)
def test_invalid_typed_block_has_no_task_run_or_event_side_effect(board, kind, payload):
    kb, conn, registry = board
    tid = kb.create_task(conn, title="Invalid request", initial_status="running")
    before_task = kb.get_task(conn, tid)
    before_run = kb.latest_run(conn, tid)
    before_events = kb.list_events(conn, tid)
    result = json.loads(
        registry.dispatch(
            "kanban_block",
            {
                "task_id": tid,
                "reason": "Choice",
                "kind": kind,
                "input_request": payload,
            },
        )
    )
    assert "error" in result, result
    assert kb.get_task(conn, tid) == before_task
    assert kb.latest_run(conn, tid) == before_run
    assert kb.list_events(conn, tid) == before_events
