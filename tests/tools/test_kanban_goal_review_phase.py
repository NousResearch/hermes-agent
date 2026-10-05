"""Phase-aware lifecycle judging through real dispatch, parser and temp boards.

The auxiliary transport below is deterministic, not a model-understanding test.
It rejects a review-ready handoff unless the real judge prompt identifies that
phase, and checks that the full original task reaches the model boundary.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def phase_worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    (home / "profiles" / "verifier").mkdir(parents=True)
    title = "Implement a bounded parser, then obtain independent review"
    body = (
        "Implement the parser, self-review it, and pass scripts/check_parser.sh.\n"
        + "Context: preserve the parser's documented input boundaries.\n" * 45
        + "Mandatory independent review by verifier follows implementation. "
        "The formal review handoff must be recorded before final completion. "
        "Release/deployment belongs to a later owner after approval. "
        "Implementation acceptance: malformed inputs must be rejected."
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, body=body, assignee="builder", goal_mode=True)
        task = kb.get_task(conn, tid)
        assert task is not None
        workspace = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, tid, workspace)
        claimed = kb.claim_task(conn, tid, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    artifact = workspace / "verification.txt"
    artifact.write_text("Synthetic fixture evidence, not a real parser verification.\n")
    return tid, f"{title}\n\n{body}", artifact


def _snapshot(tid):
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        return (
            task, kb.list_runs(conn, tid),
            kb.list_events(conn, tid), kb.list_attachments(conn, tid),
        )


@pytest.mark.parametrize("surface", ["tool", "cli"])
@pytest.mark.parametrize("action", ["kanban_request_review", "kanban_complete"])
@pytest.mark.parametrize("evidence", [
    "Implemented parser with malformed-input rejection. Self-review complete. "
    "scripts/check_parser.sh: 12 passed, exit 0. Independent review pending.",
    "Parser implementation incomplete; malformed-input handling is missing.",
    "Implemented parser. Self-review complete. scripts/check_parser.sh: 1 failed, exit 1.",
    "Implemented parser. scripts/check_parser.sh: 12 passed, exit 0. Self-review pending.",
], ids=["ready", "incomplete", "failed-check", "no-self-review"])
def test_lifecycle_judges_the_requested_phase(phase_worker, monkeypatch, surface, action, evidence):
    import agent.auxiliary_client as auxiliary
    from hermes_cli import kanban as cli
    from tools import kanban_tools  # noqa: F401 - registers the real handlers
    from tools.registry import registry

    tid, original_goal, artifact = phase_worker
    before = _snapshot(tid)
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        system, user = [message["content"] for message in kwargs["messages"]]
        ready = all(text in user for text in (
            "Implemented parser", "Self-review complete", "12 passed, exit 0",
        ))
        review_phase = "Phase: review-readiness" in system
        verdict = "done" if review_phase and ready else "continue"
        reason = "implementation ready" if verdict == "done" else "required work remains"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"verdict": verdict, "reason": reason})))])

    monkeypatch.setattr(auxiliary, "get_text_auxiliary_client", lambda purpose: (object(), "fixture"))
    monkeypatch.setattr(auxiliary, "call_llm", transport)
    # kanban_request_review validates the reviewer against installed profiles before the goal
    # gate runs; this sandbox has none, so stub the lookup and let the gate be reached.
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(profiles, "list_profile_names", lambda: ["verifier"])
    metadata = {"artifacts": [str(artifact)]}
    if surface == "tool":
        args = {"summary": evidence, "metadata": metadata}
        if action == "kanban_request_review":
            args["reviewer"] = "verifier"
        raw = registry.dispatch(action, args)
        result = json.loads(raw) if isinstance(raw, str) else raw
        accepted = result.get("ok") is True
    else:
        verb = "request-review" if action == "kanban_request_review" else "complete"
        command = [verb, tid, "--summary", evidence, "--metadata", json.dumps(metadata)]
        if action == "kanban_request_review":
            command += ["--reviewer", "verifier"]
        result = cli.run_slash(shlex.join(command))
        accepted = "Requested review" in result or "Completed" in result

    expected = action == "kanban_request_review" and "malformed-input rejection" in evidence
    assert accepted is expected, result
    assert len(calls) == 1
    system, user = [message["content"] for message in calls[0]["messages"]]
    assert f"Lifecycle action: {action}" in system
    assert original_goal in user
    assert evidence in user
    after = _snapshot(tid)
    assert artifact.is_file()
    if not expected:
        assert after == before
        return
    task, runs, events, attachments = after
    assert (task.status, task.assignee, task.current_run_id) == ("review", "verifier", None)
    assert task.goal_mode is True
    assert (task.title, task.body) == (before[0].title, before[0].body)
    assert (runs[-1].profile, runs[-1].outcome, runs[-1].summary) == (
        "builder", "review_requested", evidence,
    )
    requested = [event for event in events if event.kind == "review_requested"]
    assert len(requested) == 1
    event = requested[0]
    assert event.payload is not None
    assert event.run_id == before[0].current_run_id
    assert (event.payload["implementer"], event.payload["reviewer"]) == ("builder", "verifier")
    assert [item.stored_path for item in attachments] == event.payload["artifacts"]
    assert len(attachments) == 1
    assert Path(attachments[0].stored_path) != artifact
    assert all(Path(path).read_bytes() == artifact.read_bytes() for path in event.payload["artifacts"])


@pytest.mark.parametrize("guard", ["unknown-reviewer", "stale-run", "missing-artifact"])
def test_positive_review_judgment_cannot_bypass_transition_guards(phase_worker, monkeypatch, guard):
    from tools import kanban_tools
    from tools.registry import registry

    tid, _, artifact = phase_worker
    monkeypatch.setattr(kanban_tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(kanban_tools, "judge_goal", lambda **kwargs: (
        "done", "implementation ready", False, None, False,
    ))
    args = {"summary": "Ready for review", "reviewer": "verifier", "artifacts": [str(artifact)]}
    if guard == "unknown-reviewer":
        args["reviewer"] = "not-installed"
    elif guard == "stale-run":
        run_id = _snapshot(tid)[0].current_run_id
        assert run_id is not None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1))
    else:
        args["artifacts"].append(str(artifact.with_name("missing.txt")))
    before = _snapshot(tid)
    raw = registry.dispatch("kanban_request_review", args)
    result = json.loads(raw) if isinstance(raw, str) else raw
    assert result.get("error"), result
    assert _snapshot(tid) == before
    assert artifact.is_file()
    attachment_dir = kb.task_attachments_dir(tid)
    assert not attachment_dir.exists() or not any(attachment_dir.iterdir())
