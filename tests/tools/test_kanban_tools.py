"""Tests for the Kanban tool surface (tools/kanban_tools.py).

Verifies:
  - Tools are gated on HERMES_KANBAN_TASK: a normal chat session sees
    zero kanban tools in its schema; a worker session sees the kanban set.
  - Each handler's happy path.
  - Error paths (missing required args, bad metadata type, etc).
"""
from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_kanban_tools_hidden_without_env_var(monkeypatch, tmp_path):
    """Normal `hermes chat` sessions (no HERMES_KANBAN_TASK) must have
    zero kanban_* tools in their schema."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    assert kanban == set(), (
        f"kanban tools leaked into normal chat schema: {kanban}"
    )


def test_kanban_tools_visible_with_env_var(monkeypatch, tmp_path):
    """Worker sessions get task lifecycle tools, not board-routing tools."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    expected = {
        "kanban_show", "kanban_complete", "kanban_block", "kanban_heartbeat",
        "kanban_comment", "kanban_create", "kanban_link",
    }
    assert kanban == expected, f"expected {expected}, got {kanban}"


def test_kanban_worker_env_overrides_profile_toolset_filter(monkeypatch, tmp_path):
    """Dispatcher-spawned workers must get lifecycle tools even when the
    assignee profile restricts enabled toolsets and does not list kanban.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    schema = get_tool_definitions(
        enabled_toolsets=["terminal"],
        quiet_mode=True,
    )
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_show" in names
    assert "kanban_complete" in names
    assert "kanban_block" in names
    assert "kanban_list" not in names


def test_worker_with_kanban_toolset_still_hides_board_routing(monkeypatch, tmp_path):
    """Task scope wins over profile config for board-routing tools.

    Even if a worker process happens to also have ``toolsets: [kanban]``
    in its config, the HERMES_KANBAN_TASK env var means it's a focused
    worker and must not see kanban_list / kanban_unblock.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    assert {
        "kanban_list",
        "kanban_unblock",
    }.isdisjoint(kanban), (
        f"Board-routing tools leaked into worker schema: "
        f"{kanban & {'kanban_list', 'kanban_unblock'}}"
    )


def test_kanban_tools_visible_with_toolset_config(monkeypatch, tmp_path):
    """Orchestrator profiles with toolsets: [kanban] see all kanban tools."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    expected = {
        "kanban_list",
        "kanban_show", "kanban_complete", "kanban_block", "kanban_heartbeat",
        "kanban_comment", "kanban_create", "kanban_link",
        "kanban_unblock",
    }
    assert kanban == expected, f"expected {expected}, got {kanban}"


# ---------------------------------------------------------------------------
# Handler happy paths
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a worker: HERMES_HOME isolated, HERMES_KANBAN_TASK set
    after we've created the task."""
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


def test_show_defaults_to_env_task_id(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_show({})
    d = json.loads(out)
    assert "task" in d
    assert d["task"]["id"] == worker_env
    assert d["task"]["status"] == "running"
    assert "worker_context" in d
    assert "runs" in d


def test_show_explicit_task_id(worker_env):
    """Peek at a different task than the one in env."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="other task", assignee="peer")
    finally:
        conn.close()
    from tools import kanban_tools as kt
    out = kt._handle_show({"task_id": other})
    d = json.loads(out)
    assert d["task"]["id"] == other


def test_list_filters_tasks(monkeypatch, worker_env):
    """kanban_list gives orchestrators filtered board discovery."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="alpha", assignee="factory", priority=5)
        b = kb.create_task(conn, title="beta", assignee="reviewer")
        c = kb.create_task(conn, title="gamma", assignee="factory", tenant="other")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({"assignee": "factory", "status": "ready", "limit": 10})
    d = json.loads(out)
    ids = [t["id"] for t in d["tasks"]]
    assert ids == [a, c]
    assert d["count"] == 2
    assert d["tasks"][0]["title"] == "alpha"
    assert d["tasks"][0]["parent_count"] == 0
    assert b not in ids

    tenant_out = kt._handle_list({
        "assignee": "factory",
        "status": "ready",
        "tenant": "other",
    })
    tenant_ids = [t["id"] for t in json.loads(tenant_out)["tasks"]]
    assert tenant_ids == [c]


def test_list_rejects_invalid_status(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    out = kt._handle_list({"status": "not-a-state"})
    assert "status must be one of" in json.loads(out).get("error", "")


def test_list_rejects_bad_limit(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    assert json.loads(kt._handle_list({"limit": "nope"})).get("error")
    assert json.loads(kt._handle_list({"limit": 0})).get("error")


def test_list_parses_include_archived_string_false(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        live = kb.create_task(conn, title="live task", assignee="factory")
        archived = kb.create_task(conn, title="archived task", assignee="factory")
        assert kb.archive_task(conn, archived)
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({
        "assignee": "factory",
        "include_archived": "false",
    })
    ids = [t["id"] for t in json.loads(out)["tasks"]]
    assert live in ids
    assert archived not in ids


def test_list_parses_include_archived_string_true(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        live = kb.create_task(conn, title="live task", assignee="factory")
        archived = kb.create_task(conn, title="archived task", assignee="factory")
        assert kb.archive_task(conn, archived)
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({
        "assignee": "factory",
        "include_archived": "true",
    })
    ids = [t["id"] for t in json.loads(out)["tasks"]]
    assert live in ids
    assert archived in ids


def test_list_rejects_bad_include_archived(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    out = kt._handle_list({"include_archived": "sometimes"})
    assert "include_archived must be" in json.loads(out).get("error", "")


def test_complete_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({
        "summary": "got the thing done",
        "metadata": {"files": 2},
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"] == worker_env
    # Verify via kernel
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.summary == "got the thing done"
        assert run.metadata == {"files": 2}
    finally:
        conn.close()


def test_complete_metadata_round_trips_through_show(worker_env):
    """Structured completion metadata should be visible to downstream agents."""
    from tools import kanban_tools as kt

    handoff = {
        "changed_files": ["hermes_cli/kanban.py"],
        "verification": ["pytest tests/tools/test_kanban_tools.py -q"],
        "dependencies": [],
        "blocked_reason": None,
        "retry_notes": "none",
        "residual_risk": ["dashboard rendering not exercised"],
    }

    complete_out = kt._handle_complete({
        "summary": "finished with structured evidence",
        "metadata": handoff,
    })
    assert json.loads(complete_out)["ok"] is True

    show_out = kt._handle_show({"task_id": worker_env})
    shown = json.loads(show_out)
    assert shown["task"]["status"] == "done"
    assert shown["runs"][-1]["summary"] == "finished with structured evidence"
    assert shown["runs"][-1]["metadata"] == handoff


def test_complete_stamps_worker_session_id_from_env(monkeypatch, worker_env):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_SESSION_ID", "session-trusted")
    metadata = {"files": 2, "worker_session_id": "user-spoof"}

    out = kt._handle_complete({
        "summary": "done by scoped worker",
        "metadata": metadata,
    })
    assert json.loads(out)["ok"] is True
    assert metadata["worker_session_id"] == "user-spoof"

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.metadata == {
            "files": 2,
            "worker_session_id": "session-trusted",
        }
    finally:
        conn.close()


def test_complete_does_not_stamp_worker_session_id_without_scoped_task(
    monkeypatch, worker_env
):
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", "session-trusted")

    out = kt._handle_complete({
        "task_id": worker_env,
        "summary": "done outside worker scope",
        "metadata": {"files": 2, "worker_session_id": "user-provided"},
    })
    assert json.loads(out)["ok"] is True

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.metadata == {
            "files": 2,
            "worker_session_id": "user-provided",
        }
    finally:
        conn.close()


def test_complete_with_result_only(worker_env):
    """`result` alone (without summary) is accepted for legacy compat."""
    from tools import kanban_tools as kt
    out = kt._handle_complete({"result": "legacy result"})
    d = json.loads(out)
    assert d["ok"] is True


def test_complete_with_artifacts_lands_in_event_payload(worker_env):
    """``artifacts=[...]`` rides into the completed event payload so the
    gateway notifier can upload them as native attachments. See the
    kanban notifier in gateway/run.py for the consumer side."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "rendered the chart",
        "artifacts": ["/tmp/q3-revenue.png", "/tmp/q3-report.pdf"],
    })
    assert json.loads(out)["ok"] is True

    conn = kb.connect()
    try:
        events = kb.list_events(conn, worker_env)
        # Find the completion event
        completed = [e for e in events if e.kind == "completed"]
        assert len(completed) == 1
        payload = completed[0].payload or {}
        assert payload.get("artifacts") == [
            "/tmp/q3-revenue.png",
            "/tmp/q3-report.pdf",
        ]
        # And the artifacts also live on metadata for downstream workers
        run = kb.latest_run(conn, worker_env)
        assert run.metadata.get("artifacts") == [
            "/tmp/q3-revenue.png",
            "/tmp/q3-report.pdf",
        ]
    finally:
        conn.close()


def test_complete_artifacts_accepts_single_string(worker_env):
    """A bare string is auto-promoted to a single-element list for convenience."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "one chart",
        "artifacts": "/tmp/chart.png",
    })
    assert json.loads(out)["ok"] is True

    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.metadata.get("artifacts") == ["/tmp/chart.png"]
    finally:
        conn.close()


def test_complete_artifacts_merges_with_explicit_metadata_field(worker_env):
    """If the worker passes metadata.artifacts AND the top-level artifacts
    param, merge the two without duplicates."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "merged",
        "metadata": {"artifacts": ["/tmp/a.png"], "other": "fact"},
        "artifacts": ["/tmp/b.pdf", "/tmp/a.png"],
    })
    assert json.loads(out)["ok"] is True

    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        # Order: existing entries first, then new ones, deduplicated.
        assert run.metadata.get("artifacts") == ["/tmp/a.png", "/tmp/b.pdf"]
        assert run.metadata.get("other") == "fact"
    finally:
        conn.close()


def test_complete_rejects_non_list_artifacts(worker_env):
    """Non-list, non-string artifacts should be rejected with a clear error."""
    from tools import kanban_tools as kt
    out = kt._handle_complete({
        "summary": "bad shape",
        "artifacts": {"not": "a list"},
    })
    err = json.loads(out).get("error", "")
    assert "artifacts must be a list" in err


def test_complete_rejects_no_handoff(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({})
    assert json.loads(out).get("error"), "should have errored"


def test_complete_rejects_non_dict_metadata(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({"summary": "x", "metadata": [1, 2, 3]})
    assert json.loads(out).get("error")


def test_complete_phantom_card_message_advertises_retry(worker_env):
    """A phantom-card rejection must surface a tool_error that explicitly
    tells the worker the task is still in-flight and how to retry — the
    worker has no other channel to discover that. Regression for #22923,
    where the previous wording read like a terminal failure and workers
    routinely abandoned the run instead of trying again.
    """
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "oops claimed a phantom",
        "created_cards": ["t_phantomdeadbeef"],
    })
    err = json.loads(out).get("error", "")
    assert err, f"expected an error, got {out!r}"
    # Phantom id surfaced verbatim.
    assert "t_phantomdeadbeef" in err
    # The retry-is-supported phrasing — these are the literal cues a
    # worker reads to decide whether to retry vs block/abandon. If a
    # future change rewords the message, these checks will catch the
    # regression. See #22923 for the failure mode.
    assert "still in-flight" in err
    assert "Retry kanban_complete" in err
    assert "created_cards=[]" in err

    # Critically: the task is genuinely still in-flight — the gate
    # rejection did not mutate state, so the worker's retry can land.
    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "running"
    finally:
        conn.close()


def test_complete_retry_with_empty_created_cards_succeeds(worker_env):
    """After a phantom rejection, retrying kanban_complete with
    created_cards=[] (the documented escape hatch) must complete the
    task. Regression for #22923."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Hit the gate first.
    rejected = json.loads(kt._handle_complete({
        "summary": "oops",
        "created_cards": ["t_phantomdeadbeef"],
    }))
    assert rejected.get("error")

    # Retry with the escape hatch.
    ok = json.loads(kt._handle_complete({
        "summary": "retry without claims",
        "created_cards": [],
    }))
    assert ok.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_complete_retry_with_corrected_created_cards_succeeds(worker_env):
    """After a phantom rejection, retrying kanban_complete with a
    corrected created_cards list (phantom ids removed) must complete the
    task. Regression for #22923."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Create a real child via the tool so it gets the worker-profile
    # attribution the gate trusts.
    child = json.loads(kt._handle_create({
        "title": "real child", "assignee": "peer",
    }))
    assert child["ok"]
    real_id = child["task_id"]

    # First attempt mixes real + phantom — gate rejects.
    rejected = json.loads(kt._handle_complete({
        "summary": "oops",
        "created_cards": [real_id, "t_phantomdeadbeef"],
    }))
    assert rejected.get("error")
    assert "t_phantomdeadbeef" in rejected["error"]

    # Retry with corrected list.
    ok = json.loads(kt._handle_complete({
        "summary": "retry with corrected list",
        "created_cards": [real_id],
    }))
    assert ok.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_block_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_block({"reason": "need clarification"})
    d = json.loads(out)
    assert d["ok"] is True
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "blocked"
    finally:
        conn.close()


def test_block_rejects_empty_reason(worker_env):
    from tools import kanban_tools as kt
    for bad in ["", "   ", None]:
        out = kt._handle_block({"reason": bad})
        assert json.loads(out).get("error")


def test_heartbeat_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({"note": "progress"})
    d = json.loads(out)
    assert d["ok"] is True


def test_heartbeat_without_note(worker_env):
    """note is optional."""
    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({})
    d = json.loads(out)
    assert d["ok"] is True


def test_heartbeat_extends_claim_expires(worker_env):
    """The kanban_heartbeat tool MUST extend claim_expires, not just
    update last_heartbeat_at — otherwise long-running workers loop the
    heartbeat tool diligently and still get reclaimed by
    release_stale_claims at DEFAULT_CLAIM_TTL_SECONDS.

    Regression test for the bug where _handle_heartbeat called
    heartbeat_worker but never heartbeat_claim, so claim_expires sat
    static while last_heartbeat_at advanced.
    """
    import time as _time
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Rewind claim_expires into the past so any forward movement is
    # unambiguous (avoids time.sleep flakiness).
    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (1, worker_env),
        )
        conn.commit()
        before = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()
    assert before == 1

    out = kt._handle_heartbeat({"note": "still alive"})
    assert json.loads(out).get("ok") is True

    conn = kb.connect()
    try:
        after = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()

    now = int(_time.time())
    # claim_expires should be roughly now + DEFAULT_CLAIM_TTL_SECONDS.
    # We assert a generous floor (now + half the default TTL) to keep the
    # test stable against future TTL changes.
    assert after > before, (
        f"claim_expires did not advance ({before} -> {after}); workers "
        f"would be reclaimed at TTL despite heartbeating"
    )
    assert after >= now + (kb.DEFAULT_CLAIM_TTL_SECONDS // 2), (
        f"claim_expires={after} is suspiciously close to now={now}; "
        f"expected at least now + {kb.DEFAULT_CLAIM_TTL_SECONDS // 2}"
    )


def test_comment_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env,
        "body": "hello thread",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["comment_id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        assert len(comments) == 1
        # Author defaults to HERMES_PROFILE env we set in the fixture
        assert comments[0].author == "test-worker"
        assert comments[0].body == "hello thread"
    finally:
        conn.close()


def test_comment_rejects_empty_body(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_comment({"task_id": worker_env, "body": "   "})
    assert json.loads(out).get("error")


def test_comment_ignores_caller_supplied_author(worker_env):
    """``args["author"]`` is no longer honored — the author is always
    derived from ``HERMES_PROFILE`` so a worker can't forge a comment
    under an authoritative-looking name like ``hermes-system`` and
    poison the next worker's prompt context. Cross-task commenting
    itself remains unrestricted (see #19713); only the author override
    is removed.
    """
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env, "body": "hi", "author": "hermes-system",
    })
    assert json.loads(out)["ok"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        # Author comes from HERMES_PROFILE in the fixture, not the
        # caller-supplied "hermes-system" override.
        assert comments[0].author == "test-worker"
    finally:
        conn.close()


def test_comment_schema_omits_author_override():
    """The ``author`` property must not appear on KANBAN_COMMENT_SCHEMA;
    exposing it to the LLM would re-introduce the forgery surface this
    handler is hardened against.
    """
    from tools.kanban_tools import KANBAN_COMMENT_SCHEMA
    props = KANBAN_COMMENT_SCHEMA["parameters"]["properties"]
    assert "author" not in props


def test_create_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "child task",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"]
    assert d["status"] == "todo"  # parent isn't done yet
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        child = kb.get_task(conn, d["task_id"])
        assert child.title == "child task"
        assert child.assignee == "peer"
    finally:
        conn.close()


def test_create_schema_exposes_model_override():
    from tools.kanban_tools import KANBAN_CREATE_SCHEMA

    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert "model_override" in props
    assert props["model_override"]["type"] == "string"


def test_create_persists_model_override(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb

    out = kt._handle_create({
        "title": "cheap worker",
        "assignee": "peer",
        "model_override": "deepseek/deepseek-v4-flash:free",
    })
    d = json.loads(out)
    assert d["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, d["task_id"])
    assert task is not None
    assert task.model_override == "deepseek/deepseek-v4-flash:free"


def test_create_blank_model_override_uses_profile_default(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb

    out = kt._handle_create({
        "title": "default model",
        "assignee": "peer",
        "model_override": "   ",
    })
    d = json.loads(out)
    assert d["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, d["task_id"])
    assert task is not None
    assert task.model_override is None


def test_create_stamps_session_id_from_env(monkeypatch, worker_env):
    """When the agent loop runs under ACP, the server propagates the
    originating chat session id via HERMES_SESSION_ID. ``kanban_create``
    reads it and stamps the new task so clients can render a per-session
    board (issue: ACP session linkage on kanban tasks)."""
    monkeypatch.setenv("HERMES_SESSION_ID", "acp-sess-abc")
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "from chat",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    conn = kb.connect()
    try:
        new_task = kb.get_task(conn, d["task_id"])
        assert new_task.session_id == "acp-sess-abc"
    finally:
        conn.close()


def test_create_session_id_arg_overrides_env(monkeypatch, worker_env):
    """An explicit ``session_id`` arg from the model wins over the env
    propagation. Edge case but exercised: a tool call could carry a
    different session id (e.g. cross-session linking) and the explicit
    arg should not be silently overwritten."""
    monkeypatch.setenv("HERMES_SESSION_ID", "from-env")
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "explicit override",
        "assignee": "peer",
        "parents": [worker_env],
        "session_id": "explicit-arg",
    })
    d = json.loads(out)
    assert d["ok"] is True
    conn = kb.connect()
    try:
        new_task = kb.get_task(conn, d["task_id"])
        assert new_task.session_id == "explicit-arg"
    finally:
        conn.close()


def test_create_session_id_absent_when_env_unset(monkeypatch, worker_env):
    """No env var, no arg → session_id stays NULL. Important for backwards
    compatibility: pre-ACP-propagation hosts and CLI-driven creates must
    not accidentally inherit a stale id."""
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "no session",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    conn = kb.connect()
    try:
        new_task = kb.get_task(conn, d["task_id"])
        assert new_task.session_id is None
    finally:
        conn.close()


def test_create_rejects_no_title(worker_env):
    from tools import kanban_tools as kt
    assert json.loads(kt._handle_create({"assignee": "x"})).get("error")
    assert json.loads(kt._handle_create({"title": "   ", "assignee": "x"})).get("error")


def test_create_rejects_no_assignee(worker_env):
    from tools import kanban_tools as kt
    assert json.loads(kt._handle_create({"title": "t"})).get("error")


def test_create_rejects_non_list_parents(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({"title": "t", "assignee": "a", "parents": 42})
    assert json.loads(out).get("error")


def test_create_parses_triage_string_false(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "not triage",
        "assignee": "peer",
        "triage": "false",
    })
    d = json.loads(out)
    assert d["ok"] is True
    conn = kb.connect()
    try:
        task = kb.get_task(conn, d["task_id"])
        assert task.status == "ready"
    finally:
        conn.close()


def test_create_parses_triage_string_true(worker_env):
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "needs triage",
        "assignee": "peer",
        "triage": "true",
    })
    d = json.loads(out)
    assert d["ok"] is True
    conn = kb.connect()
    try:
        task = kb.get_task(conn, d["task_id"])
        assert task.status == "triage"
    finally:
        conn.close()


def test_create_rejects_bad_triage(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "bad triage",
        "assignee": "peer",
        "triage": "sometimes",
    })
    assert "triage must be" in json.loads(out).get("error", "")


def test_create_accepts_string_parent(worker_env):
    """Convenience: a single parent id as string is coerced to [id]."""
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "t", "assignee": "a", "parents": worker_env,
    })
    assert json.loads(out)["ok"]


def test_create_accepts_skills_list(worker_env):
    """Tool writes the per-task skills through to the kernel."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "skilled",
        "assignee": "linguist",
        "skills": ["translation", "github-code-review"],
    })
    d = json.loads(out)
    assert d["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, d["task_id"])
    assert task.skills == ["translation", "github-code-review"]


def test_create_accepts_skills_string(worker_env):
    """Convenience: a single skill name as string is coerced to [name]."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    out = kt._handle_create({
        "title": "one-skill",
        "assignee": "a",
        "skills": "translation",
    })
    d = json.loads(out)
    assert d["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, d["task_id"])
    assert task.skills == ["translation"]


def test_create_rejects_non_list_skills(worker_env):
    """skills: 42 must be rejected, not silently dropped."""
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "t", "assignee": "a", "skills": 42,
    })
    assert json.loads(out).get("error")


def test_link_happy_path(worker_env):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="A", assignee="x")
        b = kb.create_task(conn, title="B", assignee="x")
    finally:
        conn.close()
    from tools import kanban_tools as kt
    out = kt._handle_link({"parent_id": a, "child_id": b})
    d = json.loads(out)
    assert d["ok"] is True


def test_link_rejects_self_reference(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_link({"parent_id": worker_env, "child_id": worker_env})
    assert json.loads(out).get("error")


def test_link_rejects_missing_args(worker_env):
    from tools import kanban_tools as kt
    assert json.loads(kt._handle_link({"parent_id": "x"})).get("error")
    assert json.loads(kt._handle_link({"child_id": "y"})).get("error")


def test_link_rejects_cycle(worker_env):
    """A → B, then try to link B → A."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="A", assignee="x")
        b = kb.create_task(conn, title="B", assignee="x", parents=[a])
    finally:
        conn.close()
    from tools import kanban_tools as kt
    out = kt._handle_link({"parent_id": b, "child_id": a})
    assert json.loads(out).get("error")


def test_unblock_happy_path(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="blocked", assignee="worker")
        kb.block_task(conn, tid, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": tid})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "ready"

    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_unblock_rejects_non_blocked_task(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": worker_env})
    assert json.loads(out).get("error")


def test_worker_lifecycle_through_tools(worker_env):
    """Drive the full claim -> heartbeat -> comment -> complete lifecycle
    exclusively through the tools, then verify the DB state matches what
    the dispatcher/notifier expect."""
    from tools import kanban_tools as kt

    # 1. show — worker orientation
    show = json.loads(kt._handle_show({}))
    assert show["task"]["id"] == worker_env

    # 2. heartbeat during long op
    assert json.loads(kt._handle_heartbeat({"note": "warming up"}))["ok"]

    # 3. comment for a future peer
    assert json.loads(kt._handle_comment({
        "task_id": worker_env,
        "body": "note: using stdlib sqlite3 bindings",
    }))["ok"]

    # 4. spawn a child task for follow-up
    child_out = json.loads(kt._handle_create({
        "title": "write integration test",
        "assignee": "qa",
        "parents": [worker_env],
    }))
    assert child_out["ok"]

    # 5. complete with structured handoff
    comp = json.loads(kt._handle_complete({
        "summary": "implemented + spawned QA follow-up",
        "metadata": {"child_task": child_out["task_id"]},
    }))
    assert comp["ok"]

    # Verify final state
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        parent = kb.get_task(conn, worker_env)
        assert parent.status == "done"
        assert parent.current_run_id is None
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.metadata == {"child_task": child_out["task_id"]}
        # Child is todo (parent just finished, but recompute_ready may
        # have promoted it — complete_task runs recompute internally).
        child = kb.get_task(conn, child_out["task_id"])
        assert child.status == "ready", (
            f"child should be ready after parent done, got {child.status}"
        )
        # Comment is visible
        assert len(kb.list_comments(conn, worker_env)) == 1
        # Heartbeat event recorded
        hb = [e for e in kb.list_events(conn, worker_env) if e.kind == "heartbeat"]
        assert len(hb) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# System-prompt guidance injection
# ---------------------------------------------------------------------------

def test_kanban_guidance_not_in_normal_prompt(monkeypatch, tmp_path):
    """A normal chat session (no HERMES_KANBAN_TASK) must NOT have
    KANBAN_GUIDANCE in its system prompt."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from tools.registry import invalidate_check_fn_cache
    from model_tools import _clear_tool_defs_cache
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()

    from run_agent import AIAgent
    a = AIAgent(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    prompt = a._build_system_prompt()
    assert "You are a Kanban worker" not in prompt
    assert "kanban_show()" not in prompt


def test_kanban_guidance_in_worker_prompt(monkeypatch, tmp_path):
    """A worker session (HERMES_KANBAN_TASK set) MUST have the full
    lifecycle guidance in its system prompt."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from tools.registry import invalidate_check_fn_cache
    from model_tools import _clear_tool_defs_cache
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()

    from run_agent import AIAgent
    a = AIAgent(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    prompt = a._build_system_prompt()
    # Header phrase (identity-free — SOUL.md owns identity, layer 3 is protocol)
    assert "Kanban task execution protocol" in prompt
    # Lifecycle signals
    assert "kanban_show()" in prompt
    assert "kanban_complete" in prompt
    assert "kanban_block" in prompt
    assert "kanban_create" in prompt
    # Anti-shell guidance
    assert "Do not shell out" in prompt or "tools — they work" in prompt


def test_kanban_guidance_prompt_size_bounded(monkeypatch, tmp_path):
    """Sanity: the guidance block is under 4 KB so it doesn't blow
    up the cached prompt."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from agent.prompt_builder import KANBAN_GUIDANCE
    assert 1_500 < len(KANBAN_GUIDANCE) < 4_096, (
        f"KANBAN_GUIDANCE is {len(KANBAN_GUIDANCE)} chars — too short (missing?) or too long"
    )


# ---------------------------------------------------------------------------
# Worker task-ownership enforcement (regression tests for #19534)
# ---------------------------------------------------------------------------
#
# A worker process has HERMES_KANBAN_TASK set to its own task id. The
# destructive tools (kanban_complete, kanban_block, kanban_heartbeat,
# kanban_unblock) must refuse to operate
# on any OTHER task id, even if the caller supplies an explicit `task_id`
# argument. Workers legitimately call kanban_show / kanban_list /
# kanban_comment / kanban_create / kanban_link on other tasks, so those
# are unrestricted.
#
# Orchestrator profiles (no HERMES_KANBAN_TASK in env) are intentionally
# exempt — their job is routing, and they sometimes close out child
# tasks on behalf of the child.


def test_worker_complete_rejects_foreign_task_id(worker_env):
    """A worker cannot complete a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": other, "summary": "HIJACK"})
    d = json.loads(out)
    assert d.get("ok") is not True
    assert "refusing to mutate" in d.get("error", "")

    # Sibling task must be untouched.
    conn = kb.connect()
    try:
        assert kb.get_task(conn, other).status == "ready"
    finally:
        conn.close()


def test_worker_block_rejects_foreign_task_id(worker_env):
    """A worker cannot block a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_block({"task_id": other, "reason": "evil"})
    d = json.loads(out)
    assert "refusing to mutate" in d.get("error", "")

    conn = kb.connect()
    try:
        assert kb.get_task(conn, other).status == "ready"
    finally:
        conn.close()


def test_worker_heartbeat_rejects_foreign_task_id(worker_env):
    """A worker cannot heartbeat a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        # Put sibling in running state so heartbeat would otherwise succeed.
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({"task_id": other})
    d = json.loads(out)
    assert "refusing to mutate" in d.get("error", "")


def test_worker_can_comment_on_foreign_task(worker_env):
    """Cross-task commenting must remain unrestricted (#19713 policy).

    The author-forgery hardening removed args['author'] but deliberately
    did NOT add an ownership gate to kanban_comment — comments are the
    documented handoff channel between tasks. This test pins that policy
    so a future change accidentally adding ``_enforce_worker_task_ownership``
    to ``_handle_comment`` would fail CI immediately.
    """
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": other,
        "body": "handoff: see prior findings before starting",
    })
    d = json.loads(out)
    assert d.get("ok") is True, f"cross-task comment must succeed: {d}"

    # The comment lands on the foreign task, attributed to the worker's
    # HERMES_PROFILE — never to a caller-controlled string.
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, other)
        assert len(comments) == 1
        assert comments[0].author == "test-worker"
        assert comments[0].body.startswith("handoff:")
    finally:
        conn.close()


def test_worker_unblock_rejects_foreign_task_id(worker_env):
    """A worker cannot unblock any task — kanban_unblock is orchestrator-only.

    The check fires before the per-task ownership check, so the error
    surface is the orchestrator-only refusal rather than the
    cross-task-ownership refusal. Either is fine — the property we're
    pinning is "worker cannot mutate foreign task via kanban_unblock".
    """
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="blocked sibling", assignee="peer")
        kb.block_task(conn, other, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": other})
    d = json.loads(out)
    err = d.get("error", "")
    assert "orchestrator-only" in err or "refusing to mutate" in err, (
        f"expected worker-rejection error, got {err}"
    )

    conn = kb.connect()
    try:
        assert kb.get_task(conn, other).status == "blocked"
    finally:
        conn.close()


def test_worker_complete_own_task_still_works(worker_env):
    """The ownership check doesn't break the normal own-task happy path."""
    from tools import kanban_tools as kt
    # Both implicit (no task_id arg) and explicit (matching env) must work.
    out = kt._handle_complete({"task_id": worker_env, "summary": "explicit own"})
    d = json.loads(out)
    assert d.get("ok") is True and d.get("task_id") == worker_env


def test_worker_complete_rejects_stale_run_id(worker_env, monkeypatch):
    """A retried worker cannot complete the task using an old run token."""
    from hermes_cli import kanban_db as kb
    import hermes_cli.kanban_db as _kb

    conn = kb.connect()
    try:
        run1 = kb.latest_run(conn, worker_env)
        kb._set_worker_pid(conn, worker_env, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [worker_env]

        kb.claim_task(conn, worker_env)
        run2 = kb.latest_run(conn, worker_env)
        assert run2.id != run1.id
    finally:
        conn.close()

    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run1.id))
    out = kt._handle_complete({"summary": "late stale completion"})
    d = json.loads(out)
    assert d.get("ok") is not True

    conn = kb.connect()
    try:
        task = kb.get_task(conn, worker_env)
        assert task.status == "running"
        assert task.current_run_id == run2.id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run2.id))
    out = kt._handle_complete({"summary": "current completion"})
    d = json.loads(out)
    assert d.get("ok") is True


def test_orchestrator_complete_any_task_allowed(monkeypatch, tmp_path):
    """Orchestrator profiles (no HERMES_KANBAN_TASK) can still complete
    any task via explicit task_id. The check only applies to workers."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="child to close out")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": tid, "summary": "orchestrator close"})
    d = json.loads(out)
    assert d.get("ok") is True and d.get("task_id") == tid


# ---------------------------------------------------------------------------
# Optional ``board`` parameter — per-call DB override
# ---------------------------------------------------------------------------
#
# The dispatcher pins the active board via HERMES_KANBAN_BOARD env var,
# but a Telegram-side orchestrator handling multiple boards needs to be
# able to route a single tool call to a specific board's DB without
# restarting Hermes. These tests pin that ``board=<slug>`` argument
# routes each handler to that board's sqlite file, and that omitting
# ``board`` preserves the legacy env-driven resolution.


@pytest.fixture
def multi_board_env(monkeypatch, tmp_path):
    """Isolated Hermes home with two distinct kanban boards seeded.

    Returns ``("default", "alt")`` slugs. The default board has one
    pre-existing task ``seed_default``; ``alt`` has ``seed_alt``. No
    HERMES_KANBAN_TASK is pinned (orchestrator context) — workers test
    the env-task case via the existing ``worker_env`` fixture.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Make sure neither HERMES_KANBAN_DB nor HERMES_KANBAN_BOARD pin a
    # board — the test is specifically about the per-call override.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    # Default board — implicit
    conn = kb.connect()
    try:
        seed_default = kb.create_task(
            conn, title="seed-default", assignee="worker-d"
        )
    finally:
        conn.close()
    # Alt board — explicit slug routes the connection to a separate DB
    conn = kb.connect(board="alt")
    try:
        seed_alt = kb.create_task(
            conn, title="seed-alt", assignee="worker-a"
        )
    finally:
        conn.close()
    return {
        "default_seed": seed_default,
        "alt_seed": seed_alt,
        "default_db": kb.kanban_db_path(),
        "alt_db": kb.kanban_db_path(board="alt"),
    }


def test_board_param_routes_create_to_alt_board(multi_board_env):
    """kanban_create with ``board="alt"`` must write into the alt board's DB,
    not the default one."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_create({
        "title": "alt-only",
        "assignee": "worker",
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    new_tid = d["task_id"]

    # Lands on alt board.
    with kb.connect(board="alt") as conn:
        assert kb.get_task(conn, new_tid).title == "alt-only"
    # Does NOT land on default board.
    with kb.connect() as conn:
        assert kb.get_task(conn, new_tid) is None


def test_board_param_routes_list_to_alt_board(multi_board_env):
    """kanban_list filters by the board parameter, not env-active."""
    from tools import kanban_tools as kt

    # Default — sees seed-default, not seed-alt.
    default_out = json.loads(kt._handle_list({}))
    default_titles = {t["title"] for t in default_out["tasks"]}
    assert "seed-default" in default_titles
    assert "seed-alt" not in default_titles

    # Alt — sees seed-alt, not seed-default.
    alt_out = json.loads(kt._handle_list({"board": "alt"}))
    alt_titles = {t["title"] for t in alt_out["tasks"]}
    assert "seed-alt" in alt_titles
    assert "seed-default" not in alt_titles


def test_board_param_routes_show_to_alt_board(multi_board_env):
    """kanban_show reads from the board parameter, not env-active.

    Tasks across boards may share ids (the id space is per-DB) but the
    seed task ids in this fixture are distinct, so a cross-board show
    must return the matching task only when board is correct.
    """
    from tools import kanban_tools as kt

    alt_seed = multi_board_env["alt_seed"]
    # Without board override, the alt task is invisible.
    bad = json.loads(kt._handle_show({"task_id": alt_seed}))
    assert "not found" in bad.get("error", "")

    # With board override, it's readable.
    good = json.loads(kt._handle_show({"task_id": alt_seed, "board": "alt"}))
    assert good["task"]["id"] == alt_seed
    assert good["task"]["title"] == "seed-alt"


def test_board_param_routes_assign_via_create_to_alt(multi_board_env):
    """Workflow test for the 'assign' UX — create with assignee on a
    specific board. (The CLI has a separate ``kanban assign`` verb; the
    MCP surface assigns at task creation time.)"""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_create({
        "title": "alt-assigned",
        "assignee": "linguist",
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True
    with kb.connect(board="alt") as conn:
        task = kb.get_task(conn, d["task_id"])
        assert task is not None
        assert task.assignee == "linguist"


def test_board_param_routes_comment_to_alt_board(multi_board_env):
    """kanban_comment routes the insert to the alt board's DB."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    alt_seed = multi_board_env["alt_seed"]
    out = kt._handle_comment({
        "task_id": alt_seed,
        "body": "alt comment",
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True

    with kb.connect(board="alt") as conn:
        comments = kb.list_comments(conn, alt_seed)
        assert len(comments) == 1
        assert comments[0].body == "alt comment"
    # Default board does not have this task at all, so no rogue comment.
    with kb.connect() as conn:
        assert kb.get_task(conn, alt_seed) is None


def test_board_param_routes_complete_to_alt_board(multi_board_env):
    """kanban_complete on the alt board closes the alt task, leaving
    the default seed untouched."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    alt_seed = multi_board_env["alt_seed"]
    # Make alt task running so complete is valid.
    with kb.connect(board="alt") as conn:
        kb.claim_task(conn, alt_seed)

    out = kt._handle_complete({
        "task_id": alt_seed,
        "summary": "alt close",
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True

    with kb.connect(board="alt") as conn:
        assert kb.get_task(conn, alt_seed).status == "done"
    # Default seed is unchanged.
    with kb.connect() as conn:
        default_seed = multi_board_env["default_seed"]
        assert kb.get_task(conn, default_seed).status == "ready"


def test_board_param_routes_block_to_alt_board(multi_board_env):
    """kanban_block targets the alt board's DB."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    alt_seed = multi_board_env["alt_seed"]
    with kb.connect(board="alt") as conn:
        kb.claim_task(conn, alt_seed)

    out = kt._handle_block({
        "task_id": alt_seed,
        "reason": "need input on alt board",
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True

    with kb.connect(board="alt") as conn:
        assert kb.get_task(conn, alt_seed).status == "blocked"


def test_board_param_routes_unblock_to_alt_board(multi_board_env):
    """kanban_unblock targets the alt board's DB."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    alt_seed = multi_board_env["alt_seed"]
    with kb.connect(board="alt") as conn:
        kb.block_task(conn, alt_seed, reason="waiting")
        assert kb.get_task(conn, alt_seed).status == "blocked"

    out = kt._handle_unblock({"task_id": alt_seed, "board": "alt"})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "ready"

    with kb.connect(board="alt") as conn:
        assert kb.get_task(conn, alt_seed).status == "ready"


def test_board_param_routes_heartbeat_to_alt_board(monkeypatch, tmp_path):
    """kanban_heartbeat targets the alt board's DB. Worker-scoped, so we
    use the worker-env style fixture inline (pinning HERMES_KANBAN_TASK
    to a task that exists in the alt board)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "alt-worker")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    # Seed the alt board with a claimed task.
    with kb.connect(board="alt") as conn:
        tid = kb.create_task(conn, title="alt hb", assignee="alt-worker")
        kb.claim_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({"note": "alive on alt", "board": "alt"})
    d = json.loads(out)
    assert d["ok"] is True

    # Heartbeat event landed in the alt DB.
    with kb.connect(board="alt") as conn:
        events = [e for e in kb.list_events(conn, tid) if e.kind == "heartbeat"]
        assert len(events) == 1


def test_board_param_routes_link_to_alt_board(multi_board_env):
    """kanban_link operates on the alt board's DB."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect(board="alt") as conn:
        a = kb.create_task(conn, title="A-alt", assignee="x")
        b = kb.create_task(conn, title="B-alt", assignee="x")

    out = kt._handle_link({
        "parent_id": a,
        "child_id": b,
        "board": "alt",
    })
    d = json.loads(out)
    assert d["ok"] is True

    with kb.connect(board="alt") as conn:
        assert b in kb.child_ids(conn, a)


def test_board_param_none_falls_back_to_env(worker_env):
    """When ``board`` is omitted or None, behaviour is unchanged from
    before this feature — calls land on whatever the env resolves to.
    Regression guard against accidentally rewiring default resolution."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_show({})  # no board, no task_id
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    out = kt._handle_show({"task_id": worker_env, "board": None})
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    # Sanity: the env-resolved path is the legacy default DB, NOT an
    # 'alt' board path. Confirms the override path was not silently
    # forced.
    assert kb.kanban_db_path() == kb.kanban_db_path(board="default")


def test_board_param_rejects_invalid_slug(multi_board_env):
    """A board slug that fails ``_normalize_board_slug`` surfaces as a
    structured tool_error rather than a 500 / unhandled exception."""
    from tools import kanban_tools as kt

    out = kt._handle_list({"board": "Has Spaces"})
    err = json.loads(out).get("error", "")
    assert "invalid board slug" in err, f"got {err!r}"


def test_board_param_in_all_schemas():
    """All nine kanban_* tool schemas must expose an optional ``board``
    parameter. This pins the contract surfaced to the LLM — adding a
    new kanban tool without ``board`` will fail CI immediately."""
    from tools import kanban_tools as kt

    schemas = [
        kt.KANBAN_SHOW_SCHEMA,
        kt.KANBAN_LIST_SCHEMA,
        kt.KANBAN_COMPLETE_SCHEMA,
        kt.KANBAN_BLOCK_SCHEMA,
        kt.KANBAN_HEARTBEAT_SCHEMA,
        kt.KANBAN_COMMENT_SCHEMA,
        kt.KANBAN_CREATE_SCHEMA,
        kt.KANBAN_UNBLOCK_SCHEMA,
        kt.KANBAN_LINK_SCHEMA,
    ]
    for schema in schemas:
        props = schema["parameters"]["properties"]
        assert "board" in props, (
            f"{schema['name']} is missing the 'board' property"
        )
        assert props["board"]["type"] == "string"
        # board is optional everywhere — never in required.
        assert "board" not in schema["parameters"].get("required", []), (
            f"{schema['name']} marks board as required; must be optional"
        )
