"""Controlled Kanban planning + lifecycle tools (kanban_graph / archive /
unlink / promote / decompose).

Covers, per the work order:
  - ``kanban_graph`` exposes parent/child statuses and is strictly read-only
  - shared archive policy: protected exactly for ``ready``/``running``/
    ``review`` (with the block/stop-first instruction), allowed for every
    other status including legacy raw values, immune to a concurrent status
    change, never auto-blocks, and its impact + review receipt tells real
    transitions / still-waiting dependents / ready follow-ups apart
  - triage promotion landing ``ready`` vs ``todo`` + gate reason
  - atomic explicit decomposition (valid, invalid, cyclic) with no LLM call
  - link + unlink validation and their actual lifecycle receipts
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- Fixtures

@pytest.fixture
def board_env(monkeypatch, tmp_path):
    """Orchestrator-context board: isolated HERMES_HOME, no HERMES_KANBAN_TASK.

    The board-level tools (archive / promote / decompose) are orchestrator-only,
    so the dispatcher-worker env must be absent; ``kanban_graph`` and
    ``kanban_unlink`` are additionally exercised in worker scope below.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _db():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    return kb, kbc


def _snapshot(conn, task_ids):
    """Everything a read-only call must not disturb: status, events, links."""
    return {
        "statuses": {
            tid: conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0]
            for tid in task_ids
        },
        "events": conn.execute(
            "SELECT task_id, kind, COUNT(*) FROM task_events GROUP BY task_id, kind"
        ).fetchall(),
        "event_total": conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
        "links": conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        ).fetchall(),
    }


# --------------------------------------------------------------------------- kanban_graph

def test_graph_exposes_parent_and_child_statuses_and_is_read_only(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        a = kb.create_task(conn, title="A", assignee="o")
        b = kb.create_task(conn, title="B", assignee="o", parents=[a])
        c = kb.create_task(conn, title="C", assignee="o", parents=[b])
        # A finishes, but readiness has NOT been recomputed yet — B stays 'todo'.
        # If kanban_graph recomputed readiness this would flip underneath it.
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (a,))
        conn.commit()
        before = _snapshot(conn, [a, b, c])

    def _boom(*_a, **_kw):
        raise AssertionError("kanban_graph must never recompute readiness")

    monkeypatch.setattr(kb, "recompute_ready", _boom)

    out = json.loads(kt._handle_graph({"task_id": b}))

    assert out["ok"] is True
    assert out["read_only"] is True
    assert out["task"] == {"id": b, "title": "B", "status": "todo"}
    assert [(p["id"], p["title"], p["status"]) for p in out["parents"]] == [(a, "A", "done")]
    assert [(ch["id"], ch["title"], ch["status"]) for ch in out["children"]] == [(c, "C", "todo")]

    with kbc.connect() as conn:
        assert _snapshot(conn, [a, b, c]) == before, "kanban_graph wrote to the board"


def test_graph_reports_leaf_and_root_shapes(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        solo = kb.create_task(conn, title="Solo", assignee="o")
        parent = kb.create_task(conn, title="P", assignee="o")
        child = kb.create_task(conn, title="Kid", assignee="o", parents=[parent])

    root_view = json.loads(kt._handle_graph({"task_id": parent}))
    assert root_view["parents"] == []
    assert [ch["id"] for ch in root_view["children"]] == [child]

    leaf_view = json.loads(kt._handle_graph({"task_id": child}))
    assert [p["id"] for p in leaf_view["parents"]] == [parent]
    assert leaf_view["children"] == []

    lonely = json.loads(kt._handle_graph({"task_id": solo}))
    assert lonely["parents"] == [] and lonely["children"] == []


def test_graph_unknown_task_is_a_structured_error(board_env):
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_graph({"task_id": "t_nope"}))
    assert out.get("ok") is not True
    assert "not found" in out["error"]


# --------------------------------------------------------------------------- kanban_archive

_ARCHIVE_REASON = "superseded duplicate — folded into the parent card"


def _blocked_task(kb, kbc, title="blocked card", parents=()):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, assignee="o", parents=list(parents))
        assert kb.claim_task(conn, tid) is not None
        assert kb.block_task(conn, tid, reason="stuck", kind="capability") is True
        assert kb.get_task(conn, tid).status == "blocked"
    return tid


def test_archive_allows_a_blocked_task(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)

    out = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))

    assert out["ok"] is True
    assert out["status"] == "archived"
    assert out["previous_status"] == "blocked"
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"


def test_archive_allows_a_done_task_with_its_impact_receipt(board_env):
    """A completed card is historical work: archivable directly, no block step,
    and the receipt still reports the real dependent transitions."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        open_gate = kb.create_task(conn, title="still open gate", assignee="o")
        target = kb.create_task(conn, title="finished work", assignee="o")
        released = kb.create_task(conn, title="released", assignee=None, parents=[target])
        still_waiting = kb.create_task(
            conn, title="still waiting", assignee="w", parents=[target, open_gate])
        # Close the card as completed WITHOUT letting readiness recompute, so
        # the receipt has genuine before/after transitions to report.
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (target,))
        conn.commit()
        assert kb.get_task(conn, target).status == "done"
        assert kb.get_task(conn, released).status == "todo"
        assert kb.get_task(conn, still_waiting).status == "todo"

    out = json.loads(kt._handle_archive({"task_id": target, "reason": _ARCHIVE_REASON}))

    assert out["ok"] is True
    assert out["status"] == "archived"
    assert out["previous_status"] == "done"
    dep = out["dependents"]

    changed = {row["id"]: row for row in dep["changed"]}
    assert set(changed) == {released}
    assert changed[released]["before"] == "todo"
    assert changed[released]["after"] == "ready"

    waiting = {row["id"]: row for row in dep["waiting"]}
    assert set(waiting) == {still_waiting}
    assert [g["id"] for g in waiting[still_waiting]["unsatisfied_parents"]] == [open_gate]
    assert "unsatisfied parent" in waiting[still_waiting]["reason"]

    ready = {row["id"]: row for row in dep["ready_followup"]}
    assert set(ready) == {released}
    assert ready[released]["needs_assignment"] is True
    assert ready[released]["changed"] is True

    with kbc.connect() as conn:
        assert kb.get_task(conn, target).status == "archived"
        assert kb.get_task(conn, released).status == "ready"
        assert kb.get_task(conn, still_waiting).status == "todo"


@pytest.mark.parametrize("status", ["ready", "running", "review"])
def test_archive_refuses_protected_states_and_never_blocks(board_env, status):
    """The shared archive policy protects exactly ``ready``/``running``/
    ``review``: the tool quotes the block/stop-first instruction and mutates
    nothing — it never parks the card either."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="work in flight", assignee="o")
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
        conn.commit()

    out = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))

    assert out.get("ok") is not True
    assert status in out["error"]
    assert "block" in out["error"]
    assert "Nothing changed" in out["error"]
    with kbc.connect() as conn:
        row = kb.get_task(conn, tid)
        assert row.status == status, "a refusal must not mutate"
        assert row.block_kind is None, "a refusal must never park the card"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (tid,),).fetchone()[0] == 0


@pytest.mark.parametrize(
    "status", ["triage", "todo", "scheduled", "blocked", "done", "completed", "not-a-status"])
def test_archive_allows_every_non_protected_state(board_env, status):
    """Anything outside the protected set archives directly — every other valid
    status plus legacy/unrecognized raw values ("don't refuse unknown")."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=f"card {status}", assignee="o", triage=True)
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
        conn.commit()

    out = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))

    assert out["ok"] is True
    assert out["status"] == "archived"
    assert out["previous_status"] == status
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (tid,),).fetchone()[0] == 1


def test_archive_policy_holds_on_the_real_registry_dispatch(board_env):
    """The eligibility rule is enforced on the real dispatch path, not just on
    the bare handler: the registry's own ``dispatch`` refuses a protected card
    and archives a finished one with the shared policy."""
    import json as _json

    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry
    kb, kbc = _db()

    with kbc.connect() as conn:
        in_flight = kb.create_task(conn, title="in flight", assignee="o")
        finished = kb.create_task(conn, title="finished", assignee="o", triage=True)
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (finished,))
        conn.commit()

    def _payload(raw):
        return raw if isinstance(raw, dict) else _json.loads(raw)

    refused = _payload(registry.dispatch(
        "kanban_archive", {"task_id": in_flight, "reason": _ARCHIVE_REASON}))
    assert refused.get("ok") is not True
    assert "block" in refused["error"]
    ok = _payload(registry.dispatch(
        "kanban_archive", {"task_id": finished, "reason": _ARCHIVE_REASON}))
    assert ok["ok"] is True and ok["previous_status"] == "done"
    with kbc.connect() as conn:
        assert kb.get_task(conn, in_flight).status == "ready"
        assert kb.get_task(conn, finished).status == "archived"


def test_archive_of_an_already_archived_task_is_a_no_op_refusal(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)

    first = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))
    assert first["ok"] is True

    second = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))
    assert second.get("ok") is not True
    assert "already archived" in second["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (tid,),
        ).fetchone()[0] == 1, "a repeat archive must not write a second event"
        # ...and the survivor is still the FIRST archive's provenance payload:
        # the no-op neither rewrites it nor adds a payload-less event.
        assert _archived_payloads(conn, tid) == [{
            "source": "kanban_archive",
            "actor": kt._persisted_identity(),
            "reason": _ARCHIVE_REASON,
        }]


@pytest.mark.parametrize("stale_status", ["ready", "running", "review"])
def test_archive_concurrent_status_change_cannot_slip_through(board_env, monkeypatch, stale_status):
    """The archive's own status pre-read claims the card is archivable while the
    row moves into a protected status right before the write.

    The guarded archive transition is the authority, so the archive loses the
    race and changes nothing — for every protected status a concurrent writer
    could have landed.
    """
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="racer", assignee="o")
        # Start archivable so the diagnostic pre-read passes; the row then moves.
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()

    real_pre_read = kb._archive_row
    seen = {"n": 0}

    def _moving_pre_read(conn, task_id):
        # The diagnostic pre-read returns the archivable snapshot; the concurrent
        # status change lands on the SAME connection right after it, so the
        # guarded UPDATE — the authority — is what must lose.
        row = real_pre_read(conn, task_id)
        if seen["n"] == 0 and task_id == tid:
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (stale_status, task_id))
        seen["n"] += 1
        return row

    monkeypatch.setattr(kb, "_archive_row", _moving_pre_read)
    out = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))
    monkeypatch.setattr(kb, "_archive_row", real_pre_read)

    assert seen["n"] >= 1, "the stale pre-read never happened"
    assert out.get("ok") is not True
    assert "changed concurrently" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == stale_status, "lost race but archived anyway"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (tid,),).fetchone()[0] == 0


def test_archive_receipt_distinguishes_transitions_waiting_and_ready(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        open_gate = kb.create_task(conn, title="still open gate", assignee="o")
        target = kb.create_task(conn, title="about to be archived", assignee="o")

        # Released: only depends on the target -> promoted when it archives.
        released = kb.create_task(conn, title="released", assignee=None, parents=[target])
        # Still waiting: depends on the target AND on an open gate.
        still_waiting = kb.create_task(
            conn, title="still waiting", assignee="w", parents=[target, open_gate])
        # Held: a human already blocked it, and it also depends on the target.
        # Block BEFORE attaching the edge — a gated card sits in 'todo' and can
        # never be claimed — then link; link_tasks only demotes 'ready', so the
        # sticky block survives the new dependency.
        held = kb.create_task(conn, title="held", assignee="w")
        assert kb.claim_task(conn, held) is not None
        assert kb.block_task(conn, held, reason="needs a human", kind="needs_input")
        kb.link_tasks(conn, target, held)
        assert kb.get_task(conn, held).status == "blocked"

        for tid in (target, released, still_waiting, held):
            assert kb.get_task(conn, tid).status != "ready" or tid == target
        assert kb.get_task(conn, released).status == "todo"
        assert kb.get_task(conn, still_waiting).status == "todo"
        assert kb.get_task(conn, held).status == "blocked"

        assert kb.block_task(conn, target, reason="stuck", kind="capability")

    out = json.loads(kt._handle_archive({"task_id": target, "reason": _ARCHIVE_REASON}))
    assert out["ok"] is True
    dep = out["dependents"]

    # 1. Dependent that ACTUALLY changed status, before -> after.
    changed = {row["id"]: row for row in dep["changed"]}
    assert set(changed) == {released}
    assert changed[released]["before"] == "todo"
    assert changed[released]["after"] == "ready"

    # 2. Dependents still waiting, each with a reason + remaining gate/hold.
    waiting = {row["id"]: row for row in dep["waiting"]}
    assert set(waiting) == {still_waiting, held}
    assert [g["id"] for g in waiting[still_waiting]["unsatisfied_parents"]] == [open_gate]
    assert waiting[still_waiting]["unsatisfied_parents"][0]["status"] in {"ready", "todo", "running"}
    assert "unsatisfied parent" in waiting[still_waiting]["reason"]
    assert waiting[held]["unsatisfied_parents"] == []
    assert waiting[held]["hold"]["kind"] == "needs_input"
    assert "held in blocked" in waiting[held]["reason"]

    # 3. Ready dependents needing assignment / dispatch follow-up.
    ready = {row["id"]: row for row in dep["ready_followup"]}
    assert set(ready) == {released}
    assert ready[released]["assignee"] is None
    assert ready[released]["needs_assignment"] is True
    assert ready[released]["changed"] is True

    with kbc.connect() as conn:
        assert kb.get_task(conn, released).status == "ready"
        assert kb.get_task(conn, still_waiting).status == "todo"
        assert kb.get_task(conn, held).status == "blocked"


def test_archive_receipt_reports_a_remaining_gate_and_the_hold_together(board_env):
    """Regression: archiving ONE parent leaves another parent unsatisfied for a
    blocked dependent. The receipt must report BOTH causes — the remaining gate
    and the block hold — never one in place of the other."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        open_gate = kb.create_task(conn, title="still open gate", assignee="o")
        target = kb.create_task(conn, title="about to be archived", assignee="o")
        # Blocked by a human hold BEFORE it is gated: a gated card sits in
        # 'todo' and can never be claimed, and link_tasks only demotes 'ready',
        # so the sticky block survives both dependency edges.
        held_gated = kb.create_task(conn, title="held and gated", assignee="w")
        assert kb.claim_task(conn, held_gated) is not None
        assert kb.block_task(conn, held_gated, reason="needs a human",
                             kind="needs_input") is True
        kb.link_tasks(conn, target, held_gated)
        kb.link_tasks(conn, open_gate, held_gated)
        assert kb.get_task(conn, held_gated).status == "blocked"
        assert kb.get_task(conn, held_gated).block_kind == "needs_input"
        assert {p for p, _ in kb.unsatisfied_parents(conn, held_gated)} == {target, open_gate}
        assert kb.block_task(conn, target, reason="stuck", kind="capability") is True

    out = json.loads(kt._handle_archive({"task_id": target, "reason": _ARCHIVE_REASON}))
    assert out["ok"] is True
    waiting = {row["id"]: row for row in out["dependents"]["waiting"]}
    assert held_gated in waiting, "the blocked dependent is still waiting"
    row = waiting[held_gated]

    # The archived parent releases its edge; the OTHER parent still gates it.
    assert row["status"] == "blocked"
    assert [g["id"] for g in row["unsatisfied_parents"]] == [open_gate]
    assert row["unsatisfied_parents"][0]["status"] in {"ready", "todo", "running"}
    # The blocked hold is exposed even though a parent gate remains.
    assert row["hold"] == {"kind": "needs_input"}
    # ...and the reason names BOTH causes, not just the first one found.
    assert "unsatisfied parent" in row["reason"]
    assert "held in blocked" in row["reason"]
    assert "needs_input" in row["reason"]

    with kbc.connect() as conn:
        # The receipt is a read of the board, not an aspiration: nothing moved.
        assert kb.get_task(conn, held_gated).status == "blocked"
        assert kb.get_task(conn, held_gated).block_kind == "needs_input"
        assert kb.get_task(conn, target).status == "archived"


def test_archive_is_hidden_from_task_workers(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    out = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))
    assert out.get("ok") is not True
    assert "orchestrator-only" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


# --- required `reason` + the provenance payload it is stored in -------------


def _dispatch(tool, args):
    """Real registry dispatch, tolerant of str or dict handler results."""
    from tools.registry import registry
    out = registry.dispatch(tool, args)
    return out if isinstance(out, dict) else json.loads(out)


def _archived_payloads(conn, tid):
    """Parsed payload of every ``archived`` event on ``tid`` (None = no payload)."""
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'archived' ORDER BY id",
        (tid,)).fetchall()
    return [json.loads(row[0]) if row[0] else None for row in rows]


def test_archive_schema_requires_task_id_and_reason(board_env):
    """The contract starts in the schema the model actually sees: both
    arguments are ``required`` and ``reason`` is a string."""
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    from tools.registry import registry

    schema = registry.get_schema("kanban_archive")
    assert schema is not None, "kanban_archive is not registered"
    params = schema["parameters"]
    assert params["required"] == ["task_id", "reason"]
    assert params["properties"]["reason"]["type"] == "string"


@pytest.mark.parametrize(
    "bad_args, label",
    [
        ({}, "missing"),
        ({"reason": None}, "null"),
        ({"reason": ""}, "empty"),
        ({"reason": "   \n\t  "}, "whitespace"),
        ({"reason": 7}, "int"),
        ({"reason": ["fold it"]}, "array"),
        ({"reason": {"why": "fold it"}}, "object"),
    ],
    ids=["missing", "null", "empty", "whitespace", "int", "array", "object"],
)
def test_archive_rejects_an_invalid_reason_before_any_mutation(board_env, bad_args, label):
    """Missing / wrong-type / blank reasons fail on the real dispatch seam with
    the card's status, its events and its runs byte-for-byte unchanged — the
    validation runs before the board is even opened."""
    from tools import kanban_tools as kt  # noqa: F401  (registers the toolset)
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)

    def _state():
        with kbc.connect() as conn:
            return {
                "status": kb.get_task(conn, tid).status,
                "events": [tuple(r) for r in conn.execute(
                    "SELECT kind, payload, run_id FROM task_events "
                    "WHERE task_id = ? ORDER BY id", (tid,))],
                "runs": [tuple(r) for r in conn.execute(
                    "SELECT id, status, outcome, summary FROM task_runs "
                    "WHERE task_id = ? ORDER BY id", (tid,))],
            }

    before = _state()
    out = _dispatch("kanban_archive", {"task_id": tid, **bad_args})
    assert out.get("ok") is not True
    assert "reason" in out["error"], f"{label}: {out}"
    assert out["error"].endswith("Nothing changed.")
    assert _state() == before, f"{label} mutated the board"
    assert before["status"] == "blocked"


def test_archive_stores_source_actor_and_reason_and_show_reads_it_back(board_env):
    """A valid call archives normally and leaves exactly one ``archived`` event
    whose payload is the {source, actor, reason} tuple: the tool id as source,
    the trusted persisted identity as actor, the caller's own wording as reason
    (preserved verbatim). ``kanban_show`` hands the payload straight back."""
    from tools import kanban_tools as kt
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)
    reason = "duplicate of t_0badf00d — folded into the parent card"

    out = _dispatch("kanban_archive", {"task_id": tid, "reason": reason})
    assert out["ok"] is True
    assert out["reason"] == reason, "the response echoes the supplied rationale"

    expected = {
        "source": "kanban_archive",
        "actor": kt._persisted_identity(),
        "reason": reason,
    }
    with kbc.connect() as conn:
        assert _archived_payloads(conn, tid) == [expected]
        # The payload rides the SAME event row, so an audit read sees one archive.
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (tid,)).fetchone()[0] == 1

    shown = _dispatch("kanban_show", {"task_id": tid})
    archived = [e for e in shown["events"] if e["kind"] == "archived"]
    assert len(archived) == 1
    assert archived[0]["payload"] == expected


def test_archive_redacts_a_secret_in_the_reason_before_storing_and_readback(board_env):
    """The required ``reason`` is agent-authored free text, so it must cross the
    SAME ``_redact`` boundary as ``kanban_block``'s reason (comment #5895467045:
    it was the one such field that skipped it).

    Checked on the DURABLE payload and on the ``kanban_show`` readback, not just
    on the response: a leaking implementation passes a round-trip assertion, so
    the assertion is on the absence of the secret — and no real secret is used,
    only a syntactically-shaped fake token.
    """
    from tools import kanban_tools as kt
    kb, kbc = _db()
    tid = _blocked_task(kb, kbc)
    secret = "ghp_" + "B" * 40
    reason = f"superseded by t_0badf00d — token was {secret}"

    out = _dispatch("kanban_archive", {"task_id": tid, "reason": reason})
    assert out["ok"] is True, out
    assert secret not in out["reason"]
    # The rationale itself survives: redaction masks credentials, not prose.
    assert "superseded by t_0badf00d" in out["reason"]
    assert out["reason"] == kt._redact(reason), "not the established redact boundary"

    with kbc.connect() as conn:
        payloads = _archived_payloads(conn, tid)
    assert len(payloads) == 1, payloads
    stored = payloads[0]
    assert isinstance(stored, dict), payloads
    assert stored["source"] == "kanban_archive"
    assert secret not in json.dumps(stored)
    assert "superseded by t_0badf00d" in stored["reason"]

    # The board's own read path hands the payload straight back to the model.
    shown = _dispatch("kanban_show", {"task_id": tid})
    archived = [e for e in shown["events"] if e["kind"] == "archived"]
    assert len(archived) == 1
    assert archived[0]["payload"] == stored
    assert secret not in json.dumps(shown)





def test_archive_refusal_writes_no_provenance_payload(board_env):
    """A refused archive (protected status) with a perfectly valid reason writes
    no ``archived`` event at all, so there is no provenance for a move that did
    not happen — and the card keeps its status."""
    kb, kbc = _db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="in flight", assignee="o")
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        conn.commit()

    out = _dispatch("kanban_archive", {"task_id": tid, "reason": _ARCHIVE_REASON})
    assert out.get("ok") is not True
    assert "Nothing changed" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert _archived_payloads(conn, tid) == []


# --------------------------------------------------------------------------- review lane in block/archive receipts


def test_block_receipt_identifies_the_same_card_review_run(board_env):
    """The review lane is same-card: the receipt names the reviewer run
    (identified from its claimed ``source_status=review`` provenance) that this
    block just closed, and reports nothing about any other card."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="subject", assignee="builder")
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer") is True
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None

    out = json.loads(kt._handle_block({"task_id": tid, "reason": "needs a decision"}))

    assert out["ok"] is True, out
    review_receipt = out["review"]
    assert review_receipt["lane"] == "same_card"
    assert review_receipt["reviewer"] == "reviewer"
    assert "reviewer_cards" not in review_receipt, "no reviewer card is inferred from the graph"
    assert review_receipt["linked_before"] == []
    assert "same-card review run" in review_receipt["note"]
    assert len(review_receipt["review_runs"]) == 1
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "blocked"
        # The reviewer run closed truthfully and the history is preserved:
        # request_review's handoff run + the reviewer's own run.
        run = kb.latest_run(conn, tid)
        assert run is not None and run.outcome == "blocked"
        assert len(kb.list_runs(conn, tid)) == 2


def test_block_does_not_cascade_to_a_linked_reviewer_card(board_env):
    """Review is same-card: blocking the reviewed work stops THIS card's own
    reviewer run and nothing else. A linked child that merely carries the
    reviewer's assignee is an ordinary dependent — reported by the receipts,
    never blocked, never stopped on this card's behalf."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reviewed work", assignee="builder")
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer") is True
        reviewer_card = kb.create_task(
            conn, title="Review reviewed work", assignee="reviewer", parents=[tid])
        bystander = kb.create_task(
            conn, title="Publish reviewed work", assignee="worker", parents=[tid])
        # The reviewer is mid-review on its own card.
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (reviewer_card,))
        conn.commit()

    out = json.loads(kt._handle_block({"task_id": tid, "reason": "blocked on scope decision"}))

    assert out["ok"] is True, out
    assert "reviewer_cards" not in out["review"]
    assert "separate reviewer card" not in out["review"]["note"]
    assert "same-card review run" in out["review"]["note"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        # Same-card only: no cascade block, no blanket descendant block.
        assert kb.get_task(conn, reviewer_card).status == "running"
        assert kb.get_task(conn, bystander).status == "todo", "no blanket descendant block"
        assert [e for e in kb.list_events(conn, tid) if e.kind == "review_lane_stopped"] == []
        # The linked children stay in the receipt's ordinary graph evidence.
        assert {c["id"] for c in out["review"]["linked_after"]} == {reviewer_card, bystander}


def test_archive_reports_reviewer_assigned_children_without_refusing(board_env):
    """Two linked cards carry the reviewer's assignee: there is no separate
    reviewer association to resolve, so the archive does not guess and does not
    refuse. It goes through and the impact receipt reports every dependent —
    none of which is blocked or otherwise mutated on this card's behalf."""
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reviewed work", assignee="builder")
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer") is True
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()
        first = kb.create_task(conn, title="Review A", assignee="reviewer", parents=[tid])
        second = kb.create_task(conn, title="Review B", assignee="reviewer", parents=[tid])

    archived = json.loads(kt._handle_archive({"task_id": tid, "reason": _ARCHIVE_REASON}))
    assert archived.get("ok") is True, archived
    assert "reviewer_cards" not in archived["review"]

    changed = {row["id"]: row for row in archived["dependents"]["changed"]}
    assert set(changed) == {first, second}
    assert changed[first]["before"] == "todo" and changed[first]["after"] == "ready"
    assert changed[second]["before"] == "todo" and changed[second]["after"] == "ready"

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"
        # Reported as dependents, never mutated into a block.
        assert kb.get_task(conn, first).status == "ready"
        assert kb.get_task(conn, second).status == "ready"
        assert [e.kind for e in kb.list_events(conn, first) if e.kind == "blocked"] == []
        assert [e.kind for e in kb.list_events(conn, second) if e.kind == "blocked"] == []


# --------------------------------------------------------------------------- kanban_promote

def test_promote_triage_with_no_unmet_parents_lands_ready(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", assignee="o", triage=True)
        assert kb.get_task(conn, tid).status == "triage"

    out = json.loads(kt._handle_promote({"task_id": tid, "reason": "cleared"}))

    assert out["ok"] is True
    assert out["status"] == "ready"
    assert out["ready"] is True
    assert out["unmet_parents"] == []
    assert "gate_reason" not in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_promote_triage_with_unmet_parent_lands_todo_and_reports_gate(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        gate_a = kb.create_task(conn, title="gate A", assignee="o")
        gate_b = kb.create_task(conn, title="gate B", assignee="o")
        tid = kb.create_task(
            conn, title="gated idea", assignee="o", triage=True, parents=[gate_a, gate_b])
        assert kb.get_task(conn, tid).status == "triage"

    out = json.loads(kt._handle_promote({"task_id": tid}))

    assert out["ok"] is True
    assert out["status"] == "todo"
    assert out["ready"] is False
    assert {g["id"] for g in out["unmet_parents"]} == {gate_a, gate_b}
    assert "unmet parent gate(s)" in out["gate_reason"]
    assert gate_a in out["gate_reason"] and gate_b in out["gate_reason"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "todo"


def test_promote_is_restricted_to_triage_and_is_no_status_setter(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        ready_id = kb.create_task(conn, title="already flowing", assignee="o")

    out = json.loads(kt._handle_promote({"task_id": ready_id}))
    assert out.get("ok") is not True
    assert "only applies to 'triage'" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, ready_id).status == "ready"

    # A direct running transition is not expressible: only triage is accepted.
    with kbc.connect() as conn:
        triage_id = kb.create_task(conn, title="idea", assignee="o", triage=True)
    refused = json.loads(kt._handle_promote({"task_id": triage_id, "status": "running"}))
    assert refused.get("ok") is not True
    with kbc.connect() as conn:
        assert kb.get_task(conn, triage_id).status in {"triage", "ready", "todo"}


# --------------------------------------------------------------------------- kanban_link / kanban_unlink

def test_link_reports_actual_demotion_and_remaining_gates(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        open_parent = kb.create_task(conn, title="open parent", assignee="o")
        done_parent = kb.create_task(conn, title="done parent", assignee="o")
        kb.complete_task(conn, done_parent, summary="done")
        child = kb.create_task(conn, title="child", assignee="o")
        assert kb.get_task(conn, child).status == "ready"

    gated = json.loads(kt._handle_link(
        {"parent_id": open_parent, "child_id": child}))
    assert gated["ok"] is True
    assert gated["gated"] is True          # a demotion really happened
    assert gated["gated_by"] == open_parent
    assert gated["previous_status"] == "ready"
    assert gated["status"] == "todo"       # read back, not assumed
    assert [g["id"] for g in gated["unsatisfied_parents"]] == [open_parent]
    assert gated["remaining_gates"] is True

    # Linking a parent that is ALREADY terminal gates nothing and demotes nothing.
    release = json.loads(kt._handle_link(
        {"parent_id": done_parent, "child_id": child}))
    assert release["ok"] is True
    assert release["gated"] is False
    assert release["gated_by"] is None
    assert release["status"] == "todo"     # unchanged by this link
    assert [g["id"] for g in release["unsatisfied_parents"]] == [open_parent]


def test_link_validation_rejects_unknown_and_self_edges(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="only", assignee="o")

    self_link = json.loads(kt._handle_link({"parent_id": tid, "child_id": tid}))
    assert self_link.get("ok") is not True

    unknown = json.loads(kt._handle_link({"parent_id": "t_missing", "child_id": tid}))
    assert unknown.get("ok") is not True
    assert "not found" in unknown["error"]

    missing_args = json.loads(kt._handle_link({"parent_id": tid}))
    assert missing_args.get("ok") is not True


def test_link_refuses_a_foreign_child_for_a_task_worker(board_env, monkeypatch):
    """A worker may only link edges whose CHILD is its own card (the same guard
    ``kanban_unlink`` applies). The foreign child here is NOT running, so the
    run-id CAS can never catch this: without the ownership check a
    prompt-injected id would gate, demote and append events to a sibling's card.
    """
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        own = kb.create_task(conn, title="own", assignee="me")
        foreign_parent = kb.create_task(conn, title="foreign parent", assignee="peer")
        foreign_child = kb.create_task(conn, title="foreign child", assignee="peer")
        assert kb.get_task(conn, foreign_child).status == "ready"
        before = _snapshot(conn, [foreign_parent, foreign_child])
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)

    out = json.loads(kt._handle_link(
        {"parent_id": foreign_parent, "child_id": foreign_child}))
    assert out.get("ok") is not True
    assert "refusing to mutate" in out["error"]
    with kbc.connect() as conn:
        assert kb.parent_ids(conn, foreign_child) == [], "no edge may be written"
        assert _snapshot(conn, [foreign_parent, foreign_child]) == before, \
            "a refused link must not demote the child or append events"

    # The own-card handoff stays intact: a worker may still gate ITS OWN card.
    own_link = json.loads(kt._handle_link(
        {"parent_id": foreign_parent, "child_id": own}))
    assert own_link["ok"] is True
    with kbc.connect() as conn:
        assert kb.parent_ids(conn, own) == [foreign_parent]


def test_unlink_releases_child_and_reports_actual_promotion(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        other_gate = kb.create_task(conn, title="other gate", assignee="o")
        parent = kb.create_task(conn, title="parent", assignee="o")
        child = kb.create_task(conn, title="child", assignee="o", parents=[parent, other_gate])
        assert kb.get_task(conn, child).status == "todo"

    # Removing ONE of two open parents releases nothing — the receipt must not
    # claim a promotion that did not happen.
    partial = json.loads(kt._handle_unlink(
        {"parent_id": parent, "child_id": child}))
    assert partial["ok"] is True
    assert partial["removed"] is True
    assert partial["previous_status"] == "todo"
    assert partial["status"] == "todo"
    assert partial["promoted"] is False
    assert partial["gated"] is True
    assert [g["id"] for g in partial["unsatisfied_parents"]] == [other_gate]

    # Removing the last open parent promotes immediately (no tick wait).
    final = json.loads(kt._handle_unlink(
        {"parent_id": other_gate, "child_id": child}))
    assert final["ok"] is True
    assert final["removed"] is True
    assert final["previous_status"] == "todo"
    assert final["status"] == "ready"
    assert final["promoted"] is True
    assert final["gated"] is False
    assert final["unsatisfied_parents"] == []


def test_unlink_validation_rejects_unknown_self_and_missing_edges(board_env):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="a", assignee="o")
        b = kb.create_task(conn, title="b", assignee="o")

    self_edge = json.loads(kt._handle_unlink({"parent_id": a, "child_id": a}))
    assert self_edge.get("ok") is not True

    unknown = json.loads(kt._handle_unlink({"parent_id": "t_missing", "child_id": b}))
    assert unknown.get("ok") is not True
    assert "not found" in unknown["error"]

    # No such edge: reported as not removed, with no invented transition.
    no_edge = json.loads(kt._handle_unlink({"parent_id": a, "child_id": b}))
    assert no_edge["ok"] is True
    assert no_edge["removed"] is False
    assert no_edge["promoted"] is False


def test_unlink_refuses_a_foreign_child_for_a_task_worker(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        own = kb.create_task(conn, title="own", assignee="me")
        foreign_parent = kb.create_task(conn, title="foreign parent", assignee="peer")
        foreign_child = kb.create_task(conn, title="foreign child", assignee="peer",
                                       parents=[foreign_parent])
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)

    out = json.loads(kt._handle_unlink(
        {"parent_id": foreign_parent, "child_id": foreign_child}))
    assert out.get("ok") is not True
    assert "refusing to mutate" in out["error"]
    with kbc.connect() as conn:
        assert kb.parent_ids(conn, foreign_child) == [foreign_parent]


def _stale_but_eligible(conn, kb, title: str) -> str:
    """A card whose row still says ``todo`` although nothing gates it.

    That is the between-ticks state a board-wide ``recompute_ready`` cleans
    up. Every normal completion path recomputes as it goes, so staleness is
    written directly here; ``unsatisfied_parents == []`` proves a board-wide
    recompute WOULD promote this card, which is what makes the side effect
    observable.
    """
    tid = kb.create_task(conn, title=title, assignee="peer")
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,))
    assert kb.get_task(conn, tid).status == "todo"
    assert kb.unsatisfied_parents(conn, tid) == []
    return tid


def _events_of(conn, tid) -> list:
    """``(kind, count)`` per event kind for one task (integer-indexed on purpose)."""
    return sorted(
        (row[0], row[1]) for row in conn.execute(
            "SELECT kind, COUNT(*) FROM task_events WHERE task_id = ? GROUP BY kind",
            (tid,)))


def test_unlink_as_a_task_worker_leaves_unrelated_tasks_untouched(board_env, monkeypatch):
    """Round-2 regression: a worker-scoped unlink may move ONLY its own card.

    ``unlink_tasks`` recomputes readiness after dropping an edge; run
    board-wide that promotes an unrelated stale-but-eligible task and appends
    its ``promoted`` event — a cross-task mutation performed by a worker that
    merely removed one of its own dependencies.
    """
    from tools import kanban_tools as kt
    kb, kbc = _db()

    with kbc.connect() as conn:
        gate = kb.create_task(conn, title="gate", assignee="peer")
        own = kb.create_task(conn, title="own", assignee="me", parents=[gate])
        unrelated = _stale_but_eligible(conn, kb, "unrelated")
        assert kb.get_task(conn, own).status == "todo"
        before_status = kb.get_task(conn, unrelated).status
        before_events = _events_of(conn, unrelated)

    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    out = json.loads(kt._handle_unlink({"parent_id": gate, "child_id": own}))

    # The affected child's own lifecycle receipt stays correct and complete.
    assert out["ok"] is True
    assert out["removed"] is True
    assert out["previous_status"] == "todo"
    assert out["status"] == "ready"
    assert out["promoted"] is True
    assert out["gated"] is False
    assert out["unsatisfied_parents"] == []

    with kbc.connect() as conn:
        # Nothing moved on the unrelated card and no event was written to it.
        assert before_status == "todo"
        assert kb.get_task(conn, unrelated).status == before_status
        assert _events_of(conn, unrelated) == before_events
        # ...while the child really was promoted, so the receipt is not a claim.
        assert kb.get_task(conn, own).status == "ready"
        assert "promoted" in [e.kind for e in kb.list_events(conn, own)]
        assert "unlinked" in [e.kind for e in kb.list_events(conn, own)]


def test_unlink_without_worker_scope_keeps_the_board_wide_recompute(board_env):
    """CLI / orchestrator / dashboard callers keep the historical behaviour.

    No ``HERMES_KANBAN_TASK`` means no worker scope: the post-unlink recompute
    stays board-wide, so the same call still cleans up a stale-but-eligible
    task (the contract ``test_unlink_tasks_triggers_recompute_ready`` pins at
    the DB layer).
    """
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        gate = kb.create_task(conn, title="gate", assignee="peer")
        own = kb.create_task(conn, title="own", assignee="me", parents=[gate])
        unrelated = _stale_but_eligible(conn, kb, "unrelated")

    out = json.loads(kt._handle_unlink({"parent_id": gate, "child_id": own}))
    assert out["ok"] is True
    assert out["removed"] is True
    assert out["promoted"] is True

    with kbc.connect() as conn:
        assert kb.get_task(conn, own).status == "ready"
        assert kb.get_task(conn, unrelated).status == "ready"
        assert "promoted" in [e.kind for e in kb.list_events(conn, unrelated)]


# --------------------------------------------------------------------------- kanban_decompose

def _patch_roster(names):
    """Pretend these profiles exist (established decompose-test convention)."""
    from types import SimpleNamespace
    from unittest.mock import patch
    fake = [
        SimpleNamespace(name=n, is_default=(i == 0), description=f"desc for {n}",
                        description_auto=False, model="m", provider="p", skill_count=1)
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name",
              return_value=names[0] if names else "default"),
    ]


def _refuse_llm(monkeypatch):
    """Any auxiliary call is a hard failure — the explicit tool must never make one."""
    import agent.auxiliary_client as aux
    from hermes_cli import kanban_decompose as decomp

    def _boom(**_kwargs):
        raise AssertionError("kanban_decompose must not invoke the auxiliary model")

    monkeypatch.setattr(aux, "call_llm", _boom)
    monkeypatch.setattr(decomp, "decompose_task", _boom)
    monkeypatch.setattr(decomp, "_call_aux", _boom)


def test_decompose_applies_graph_atomically_without_any_llm(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    _refuse_llm(monkeypatch)
    patches = _patch_roster(["owner", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="ship it", assignee="owner", triage=True)

        out = json.loads(kt._handle_decompose({
            "task_id": tid,
            "children": [
                {"title": "research", "body": "prior art", "assignee": "researcher"},
                {"title": "build", "body": "code", "assignee": "engineer", "parents": [0]},
            ],
        }))
        assert out["ok"] is True, out
        assert len(out["child_ids"]) == 2

        # Root waits on the whole graph: every child is one of its parents.
        assert out["root"]["status"] == "todo"
        assert sorted(out["root"]["parents"]) == sorted(out["child_ids"])
        assert out["status"] == "todo"
        # Established contract, with no orchestrator_profile configured: the
        # root's own assignee is preserved rather than swapped out.
        assert out["root"]["assignee"] == "owner"

        # Actual per-child states after eligibility recompute.
        by_id = {row["id"]: row for row in out["children"]}
        assert by_id[out["child_ids"][0]]["status"] == "ready"
        assert by_id[out["child_ids"][0]]["assignee"] == "researcher"
        assert by_id[out["child_ids"][1]]["status"] == "todo"
        assert by_id[out["child_ids"][1]]["parents"] == [out["child_ids"][0]]

        with kbc.connect() as conn:
            root = kb.get_task(conn, tid)
            assert root.status == "todo"
            assert sorted(kb.parent_ids(conn, tid)) == sorted(out["child_ids"])
            assert any(e.kind == "decomposed" for e in kb.list_events(conn, tid))
    finally:
        for p in patches:
            p.stop()


def test_decompose_invalid_graph_writes_nothing(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    _refuse_llm(monkeypatch)
    patches = _patch_roster(["owner", "researcher"])
    for p in patches:
        p.start()
    try:
        for label, children in [
            ("missing title", [{"assignee": "researcher"}]),
            ("parent index out of range",
             [{"title": "a", "assignee": "researcher", "parents": [7]}]),
            ("self parent",
             [{"title": "a", "assignee": "researcher", "parents": [0]}]),
            ("unknown assignee",
             [{"title": "a", "assignee": "nobody-here"}]),
            ("not a dict", ["just a string"]),
            ("empty graph", []),
        ]:
            with kbc.connect() as conn:
                tid = kb.create_task(conn, title=f"idea {label}", assignee="owner", triage=True)
                before = _snapshot(conn, [tid])
                before_children = conn.execute(
                    "SELECT COUNT(*) FROM tasks").fetchone()[0]

            out = json.loads(kt._handle_decompose({"task_id": tid, "children": children}))

            assert out.get("ok") is not True, f"{label}: expected rejection"
            with kbc.connect() as conn:
                assert _snapshot(conn, [tid]) == before, f"{label}: board was mutated"
                assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before_children
                root = kb.get_task(conn, tid)
                assert root.status == "triage", f"{label}: root left triage"
                assert kb.parent_ids(conn, tid) == []
    finally:
        for p in patches:
            p.stop()


def test_decompose_cyclic_graph_writes_nothing(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    _refuse_llm(monkeypatch)
    patches = _patch_roster(["owner", "a", "b"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="cyclic idea", assignee="owner", triage=True)
            before = _snapshot(conn, [tid])
            before_children = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        out = json.loads(kt._handle_decompose({
            "task_id": tid,
            "children": [
                {"title": "A", "assignee": "a", "parents": [1]},
                {"title": "B", "assignee": "b", "parents": [0]},
            ],
        }))

        assert out.get("ok") is not True
        assert "cyclic" in out["error"]
        with kbc.connect() as conn:
            assert _snapshot(conn, [tid]) == before
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before_children
            assert kb.get_task(conn, tid).status == "triage"
            assert kb.list_events(conn, tid) and not any(
                e.kind == "decomposed" for e in kb.list_events(conn, tid))
    finally:
        for p in patches:
            p.stop()


def test_decompose_refuses_a_non_triage_task_and_only_runs_once(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    _refuse_llm(monkeypatch)
    patches = _patch_roster(["owner", "a"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            ready_id = kb.create_task(conn, title="flowing", assignee="owner")
        refused = json.loads(kt._handle_decompose(
            {"task_id": ready_id, "children": [{"title": "x", "assignee": "a"}]}))
        assert refused.get("ok") is not True
        assert "only a 'triage' task" in refused["error"]

        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="once", assignee="owner", triage=True)
        first = json.loads(kt._handle_decompose(
            {"task_id": tid, "children": [{"title": "x", "assignee": "a"}]}))
        assert first["ok"] is True

        second = json.loads(kt._handle_decompose(
            {"task_id": tid, "children": [{"title": "y", "assignee": "a"}]}))
        assert second.get("ok") is not True
        # Once-only is enforced by the triage gate: a successful decompose
        # moves the root into the flow, so a repeat call is refused there.
        assert "only a 'triage' task" in second["error"]
        with kbc.connect() as conn:
            # Decomposition makes each child a PARENT of the root (the root
            # waits on them), so the fan-out is visible on the root's parent edge.
            assert sorted(kb.parent_ids(conn, tid)) == sorted(first["child_ids"])
            assert len(first["child_ids"]) == 1
    finally:
        for p in patches:
            p.stop()


def test_decompose_omitted_assignee_falls_back_to_the_default(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    _refuse_llm(monkeypatch)
    patches = _patch_roster(["owner", "fallback-profile"])
    for p in patches:
        p.start()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="idea", assignee="owner", triage=True)
        out = json.loads(kt._handle_decompose({
            "task_id": tid, "children": [{"title": "unrouted child"}]}))
        assert out["ok"] is True, out
        # Never a child with assignee=None (it could never be dispatched).
        assert out["children"][0]["assignee"] is not None
    finally:
        for p in patches:
            p.stop()


def test_decompose_is_hidden_from_task_workers(board_env, monkeypatch):
    from tools import kanban_tools as kt
    kb, kbc = _db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="idea", assignee="owner", triage=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    out = json.loads(kt._handle_decompose(
        {"task_id": tid, "children": [{"title": "x"}]}))
    assert out.get("ok") is not True
    assert "orchestrator-only" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
