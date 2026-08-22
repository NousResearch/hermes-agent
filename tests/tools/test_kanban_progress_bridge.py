"""Runtime → board progress/liveness bridge (tools/kanban_tools.py).

``AIAgent._touch_activity`` fires on every stream chunk, every API wait and
every retry notice.  It bridges into the kanban board so a dispatcher-spawned
worker is not reclaimed mid-flight.  The bug: that bridge renewed the *only*
lease the watchdogs looked at, so a worker that reasoned for 17K tokens with
zero tool calls held its claim forever.

These tests pin the split:

* stream/API activity  → liveness only
* tool invocation      → liveness + progress (via the tool middleware)
* ``kanban_heartbeat`` → liveness + commentary, NEVER progress: its note is
  free text the model writes, so gating the lease on it would be a spelling
  test, not a guard.

Everything runs against real imports and a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import kanban_tools as _kanban_tools

# Captured at import, BEFORE ``_reset_rate_limits`` relaxes the window to 0.0
# for the tests that drive the bridge back-to-back. A test about the limiter's
# own pacing has to use the shipped value, not the relaxed one.
_REAL_PROGRESS_WINDOW_S = _kanban_tools._AUTO_PROGRESS_MIN_INTERVAL_SECONDS


def test_limiter_sentinels_are_none_on_fresh_import():
    """Pin module defaults without reloading the shared module in this process."""
    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "from tools import kanban_tools as kt; "
        "assert kt._auto_progress_last_attempt is None; "
        "assert kt._auto_heartbeat_last_attempt is None"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """A dispatcher-spawned worker: isolated home, claimed running task,
    the env vars the dispatcher pins at spawn time."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:worker")
        run_id = kb.get_task(conn, tid).current_run_id
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:worker")

    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)
    return tid


def _reset_rate_limits(monkeypatch, kt):
    """The bridge rate-limits its DB writes; tests drive it back-to-back."""
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 0.0)


def _lease(tid):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        return conn.execute(
            "SELECT last_heartbeat_at, last_progress_at, claim_expires "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
    finally:
        conn.close()


def _rewind(tid, *, heartbeat, progress):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (heartbeat, progress, tid),
        )
        conn.execute(
            "UPDATE task_runs SET last_progress_at = ? WHERE task_id = ?",
            (progress, tid),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------

def test_stream_activity_renews_liveness_only(worker_env, monkeypatch):
    """Reproduces the masking: thousands of stream tokens keep the claim
    alive but must leave the progress lease untouched."""
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    for _ in range(10):
        kt.heartbeat_current_worker_from_env()

    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old
    assert row["last_progress_at"] == old


def test_tool_invocation_renews_progress(worker_env, monkeypatch):
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    assert kt.note_tool_progress_from_env("terminal") is True

    row = _lease(worker_env)
    assert row["last_progress_at"] > old


def test_liveness_tool_cannot_renew_its_own_progress_lease(worker_env, monkeypatch):
    """``kanban_heartbeat`` is a tool call, but its only effect is renewing
    the lease. If it counted as progress evidence the guard would be
    trivially defeatable by a model told to heartbeat while it thinks."""
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    assert kt.note_tool_progress_from_env("kanban_heartbeat") is False
    assert _lease(worker_env)["last_progress_at"] == old

    # Any other tool in the same position does count.
    assert kt.note_tool_progress_from_env("read_file") is True
    assert _lease(worker_env)["last_progress_at"] > old


def test_bridge_is_inert_outside_a_worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    assert kt.note_tool_progress_from_env("terminal") is False


def test_progress_write_is_rate_limited(worker_env, monkeypatch):
    """A tool call every few seconds for hours must not become a DB write
    every few seconds for hours."""
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)

    assert kt.note_tool_progress_from_env("terminal") is True
    assert kt.note_tool_progress_from_env("terminal") is False


@pytest.mark.parametrize("uptime", [0.0, 1.0, 3599.999])
def test_progress_limiter_admits_first_event_before_interval_uptime(
    worker_env, monkeypatch, uptime,
):
    """None is the never-attempted sentinel; every float is a real stamp.

    A worker launched in the first limiter window after boot must not lose its
    first deterministic progress event merely because monotonic uptime is
    smaller than the configured interval.
    """
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(time, "monotonic", lambda: uptime)

    assert kt.note_tool_progress_from_env("terminal") is True
    assert kt._auto_progress_last_attempt == uptime
    assert kt._auto_progress_last_attempt is not None
    assert kt.note_tool_progress_from_env("terminal") is False


@pytest.mark.parametrize("uptime", [0.0, 1.0, 3599.999])
def test_heartbeat_limiter_admits_first_event_before_interval_uptime(
    worker_env, monkeypatch, uptime,
):
    """The liveness bridge uses the same Optional sentinel contract."""
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(time, "monotonic", lambda: uptime)

    assert kt.heartbeat_current_worker_from_env() is True
    assert kt._auto_heartbeat_last_attempt == uptime
    assert kt._auto_heartbeat_last_attempt is not None
    assert kt.heartbeat_current_worker_from_env() is False


# ---------------------------------------------------------------------------
# _touch_activity — the real agent method, real bridge
# ---------------------------------------------------------------------------

def _activity_stub():
    """Minimal object carrying only what ``_touch_activity`` reads."""
    import run_agent as _ra

    class _Stub:
        session_id = ""
        _session_db = None
        _last_activity_ts = 0.0
        _last_activity_desc = ""
        _last_activity_provenance = None
        _session_activity_last_persist_mono = 0.0

    stub = _Stub()
    stub._touch_activity = _ra.AIAgent._touch_activity.__get__(stub)
    stub._persist_session_activity_if_due = (
        _ra.AIAgent._persist_session_activity_if_due.__get__(stub)
    )
    return stub


def test_touch_activity_stream_chunk_does_not_renew_progress(worker_env, monkeypatch):
    """E2E through the real ``AIAgent._touch_activity``: the exact call the
    streaming path makes on every chunk."""
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    agent = _activity_stub()
    for _ in range(20):
        agent._touch_activity("receiving stream response")
        agent._touch_activity("waiting for non-streaming API response")

    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old, "liveness must still be bridged"
    assert row["last_progress_at"] == old, (
        "stream tokens must never renew the progress lease"
    )


def test_tool_executor_notes_progress_for_a_real_tool(worker_env, monkeypatch):
    """The tool executor is where deterministic progress evidence is
    emitted; assert the real helper reaches the real board."""
    import agent.tool_executor as te
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    te._note_tool_progress("read_file")

    assert _lease(worker_env)["last_progress_at"] > old


def test_tool_executor_progress_never_raises(monkeypatch, tmp_path):
    """A bridge failure must never break the agent loop."""
    import agent.tool_executor as te
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_does_not_exist")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    te._note_tool_progress("terminal")  # must not raise


# ---------------------------------------------------------------------------
# kanban_heartbeat: liveness + commentary, NEVER progress
# ---------------------------------------------------------------------------
# The tool is the one call whose only effect is renewing a lease. Letting its
# free-text ``note`` renew progress — even gated on "new evidence" — makes the
# guard a spelling test: nothing checks the sentence against the world, so a
# worker stuck in a reasoning loop escapes by varying a string. The note stays
# useful as commentary in the event log; it just cannot move the lease.

def test_heartbeat_without_note_is_liveness_only(worker_env):
    from tools import kanban_tools as kt

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    d = json.loads(kt._handle_heartbeat({}))
    assert d.get("ok") is True
    assert d["liveness_renewed"] is True
    assert d["progress_recorded"] is False
    assert d["progress_reason"] == "liveness_only"
    assert "progress" in (d.get("hint") or "").lower()

    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old
    assert row["last_progress_at"] == old


def test_heartbeat_with_detailed_evidence_still_does_not_renew_progress(worker_env):
    """The exact shape the old guidance asked for — a concrete, specific,
    first-time note — must not move the progress lease."""
    from tools import kanban_tools as kt

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    d = json.loads(kt._handle_heartbeat(
        {"note": "wrote migrations/003_users.sql; ran pytest -q; 41 passed"}
    ))
    assert d["progress_recorded"] is False
    assert d["progress_reason"] == "liveness_only"
    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old
    assert row["last_progress_at"] == old


def test_heartbeat_alternating_notes_never_renew_progress(worker_env):
    """A/B/A/B defeats any 'is this note new?' dedupe that only remembers the
    previous note. It must not matter — no note renews progress at all."""
    from tools import kanban_tools as kt

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    for i in range(8):
        note = "compiling module A" if i % 2 == 0 else "compiling module B"
        d = json.loads(kt._handle_heartbeat({"note": note}))
        assert d["progress_recorded"] is False, note
        assert _lease(worker_env)["last_progress_at"] == old, note


def test_heartbeat_arbitrary_unique_notes_never_renew_progress(worker_env):
    """A model can trivially emit a never-repeating note. Twenty distinct,
    plausible-sounding checkpoints buy zero progress lease."""
    from tools import kanban_tools as kt

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    for i in range(20):
        d = json.loads(kt._handle_heartbeat(
            {"note": f"step {i}: analysed subsystem {i} and chose approach {i}"}
        ))
        assert d["progress_recorded"] is False, i
        assert _lease(worker_env)["last_progress_at"] == old, i
    # ...and the liveness lease did keep moving, which is the tool's job.
    assert _lease(worker_env)["last_heartbeat_at"] > old


def test_heartbeat_note_is_still_recorded_as_commentary(worker_env):
    """Removing the lease effect must not remove the operator-visible note."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    kt._handle_heartbeat({"note": "waiting on the crawl to finish"})
    conn = kb.connect()
    try:
        notes = [
            json.loads(r["payload"])["note"]
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'heartbeat' AND payload IS NOT NULL ORDER BY id",
                (worker_env,),
            )
        ]
    finally:
        conn.close()
    assert "waiting on the crawl to finish" in notes


def test_heartbeat_schema_does_not_promise_progress_from_a_note(worker_env):
    """The schema is the contract the model reads. It must not tell the model
    that writing a good enough note keeps the claim."""
    from tools import kanban_tools as kt
    desc = kt.KANBAN_HEARTBEAT_SCHEMA["description"].lower()
    note_desc = (
        kt.KANBAN_HEARTBEAT_SCHEMA["parameters"]["properties"]["note"]
        ["description"].lower()
    )
    assert "liveness" in desc
    blob = desc + " " + note_desc
    for lure in ("renews the progress", "renews progress", "evidence"):
        assert lure not in blob, lure


def test_worker_guidance_does_not_offer_notes_as_a_progress_lever(worker_env):
    """Workers do what the protocol tells them. Guidance that says a note can
    renew progress teaches the exact behaviour the guard must not reward."""
    from agent.prompt_builder import KANBAN_GUIDANCE
    g = KANBAN_GUIDANCE.lower()
    assert "progress" in g
    assert "no_progress_timeout_seconds" in g
    assert "evidence" not in g
    assert "kanban_heartbeat" in g


def test_heartbeat_tool_is_excluded_from_the_middleware_progress_signal(worker_env):
    """Belt and braces: even routed through the tool middleware, the
    lease-renewal tool is not 'a tool ran'."""
    from tools import kanban_tools as kt
    assert kt.note_tool_progress_from_env("kanban_heartbeat") is False


# ---------------------------------------------------------------------------
# Board state transitions through the worker toolset
# ---------------------------------------------------------------------------

def test_worker_comment_renews_progress(worker_env, monkeypatch):
    from tools import kanban_tools as kt

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    kt._handle_comment({"task_id": worker_env, "body": "hotspot: a.py — churn"})

    assert _lease(worker_env)["last_progress_at"] > old


# ---------------------------------------------------------------------------
# E2E: agent activity -> board leases -> dispatcher decision
# ---------------------------------------------------------------------------

def _dispatch(monkeypatch, *, alive=True):
    """Run one real dispatcher tick with a fake host-local worker process."""
    from hermes_cli import kanban_db as kb

    state = {"alive": alive}
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: state["alive"])
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda pid, lock, **kw: (
            state.update(alive=False),
            {"prev_pid": pid, "host_local": True,
             "termination_attempted": True, "terminated": True,
             "sigkill": False},
        )[1],
    )
    conn = kb.connect()
    try:
        return kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)
    finally:
        conn.close()


def _set_pid(tid, pid=4242):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb._set_worker_pid(conn, tid, pid)
    finally:
        conn.close()


def test_e2e_reasoning_only_worker_is_reclaimed(worker_env, monkeypatch):
    """The reproduced incident, end to end: a worker streams for a long time,
    calls no tool, transitions nothing — and the dispatcher takes the claim
    back with a receipt saying exactly that."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)
    _set_pid(worker_env)

    agent = _activity_stub()
    for _ in range(50):
        agent._touch_activity("receiving stream response")

    # Age the claim past the configured window. (Wall-clock is the one thing
    # a test can't wait out; everything else here is the real path.)
    from hermes_cli.config import load_config
    limit = kb.resolve_no_progress_timeout_seconds(
        (load_config().get("kanban") or {}).get("no_progress_timeout_seconds")
    )
    now = int(time.time())
    _rewind(worker_env, heartbeat=now, progress=now - limit - 60)

    result = _dispatch(monkeypatch)
    assert result.no_progress == [worker_env]

    conn = kb.connect()
    try:
        task = kb.get_task(conn, worker_env)
        assert task.status == "ready"
        assert task.consecutive_failures == 1
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (worker_env,),
        ).fetchone()
        assert run["outcome"] == "no_progress"
        meta = json.loads(run["metadata"])
        assert meta["worker_state"] == "alive"
        assert meta["liveness"] == "fresh"
        assert meta["terminated"] is True
        assert any(
            "no observable progress" in c.body
            for c in kb.list_comments(conn, worker_env)
        )
    finally:
        conn.close()


def test_e2e_worker_that_uses_a_tool_keeps_its_claim(worker_env, monkeypatch):
    """Same elapsed wall-clock, one real tool call: the claim survives."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    _reset_rate_limits(monkeypatch, kt)
    _set_pid(worker_env)

    from hermes_cli.config import load_config
    limit = kb.resolve_no_progress_timeout_seconds(
        (load_config().get("kanban") or {}).get("no_progress_timeout_seconds")
    )
    now = int(time.time())
    _rewind(worker_env, heartbeat=now, progress=now - limit - 60)

    # ... and then the worker actually does something.
    import agent.tool_executor as te
    te._note_tool_progress("terminal")

    result = _dispatch(monkeypatch)
    assert result.no_progress == []

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "running"
    finally:
        conn.close()


def test_e2e_survives_a_board_reopened_after_migration(worker_env, monkeypatch):
    """Restart path: the board is migrated in place by a later open, and the
    progress lease keeps working on the pre-existing in-flight claim."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        conn.execute("ALTER TABLE tasks DROP COLUMN last_progress_at")
        conn.execute("ALTER TABLE task_runs DROP COLUMN last_progress_at")
        conn.commit()
    finally:
        conn.close()

    # Fresh open == the restart. init_db()/connect() run the migration.
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    _reset_rate_limits(monkeypatch, kt)
    assert kt.note_tool_progress_from_env("read_file") is True

    row = _lease(worker_env)
    assert row["last_progress_at"] is not None


# ---------------------------------------------------------------------------
# Shared limiter state
# ---------------------------------------------------------------------------

def test_progress_limiter_state_is_guarded_by_a_lock(worker_env, monkeypatch):
    """The tool executor runs an in-flight ticker thread per concurrent tool
    call, and every one of them touches the module-level limiter timestamp.
    Read-compare-write without a lock lets two threads both observe the old
    value and both write, which is how a limiter silently stops limiting."""
    import threading
    from tools import kanban_tools as kt

    assert isinstance(kt._auto_progress_lock, type(threading.Lock()))
    assert isinstance(kt._auto_heartbeat_lock, type(threading.Lock()))

    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)

    admitted: list[bool] = []
    barrier = threading.Barrier(8)

    def _hit():
        barrier.wait()
        admitted.append(kt.note_tool_progress_from_env("terminal"))

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Exactly one of eight racing callers may pass the rate limiter.
    assert sum(1 for a in admitted if a) == 1, admitted


def test_progress_limiter_window_is_skewed_below_the_in_flight_cadence():
    """Equal is the one value that makes the limiter eat the ticks it paces.

    The tool executor ticks in-flight progress every
    ``_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S``. A tick only clears an equal-sized
    window if the previous stamp was this thread's own previous tick — the
    moment anything else stamps in between (a concurrent tool's start/complete
    edge tick, another tool's ticker thread) the next tick lands inside the
    window and is dropped, and the effective cadence doubles. Keeping the
    window strictly smaller means a shadowed tick still lands.
    """
    from agent import tool_executor as te

    assert (
        _REAL_PROGRESS_WINDOW_S < te._TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S
    ), (
        "the progress limiter window must stay strictly below the in-flight "
        "tick cadence"
    )


def test_a_shadowed_in_flight_tick_is_not_dropped(worker_env, monkeypatch):
    """The concrete failure the skew prevents: a foreign stamp lands shortly
    before a blocking tool's in-flight tick, and that tick must still renew
    the lease rather than wait another whole cadence."""
    from agent import tool_executor as te
    from tools import kanban_tools as kt

    cadence = te._TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S
    clock = {"t": 1000.0}
    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    # Undo the fixture's relaxation: this test is ABOUT the shipped window.
    monkeypatch.setattr(
        kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", _REAL_PROGRESS_WINDOW_S)

    import time as _time
    monkeypatch.setattr(_time, "monotonic", lambda: clock["t"])

    # A concurrent tool's edge stamp, most of a cadence into our tool's wait.
    assert kt.note_tool_progress_from_env("terminal") is True

    # Our blocking tool's in-flight tick, one full cadence after IT started.
    clock["t"] += cadence * 0.9
    assert kt.note_tool_progress_from_env("terminal") is True, (
        "a tick shadowed by a foreign stamp must still renew progress"
    )


def test_heartbeat_bridge_limiter_is_also_guarded(worker_env, monkeypatch):
    import threading
    from tools import kanban_tools as kt

    assert isinstance(kt._auto_heartbeat_lock, type(threading.Lock()))
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)

    admitted: list[bool] = []
    barrier = threading.Barrier(8)

    def _hit():
        barrier.wait()
        admitted.append(kt.heartbeat_current_worker_from_env())

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert sum(1 for a in admitted if a) == 1, admitted
