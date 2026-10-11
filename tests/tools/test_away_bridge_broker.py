"""Broker tests for the away-mode Telegram bridge (design §4.1–§4.2, §4.5–§4.7).

Uses two real SQLite connections for concurrency claims (§8.1) and a temp
HERMES_HOME per test. Lineage resolution is exercised through the real
SessionDB when available.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from tools.away_bridge_broker import (
    AlreadyAwaitingError,
    AwayBridgeBroker,
    NotArmedError,
    bridge_db_path,
    load_away_bridge_config,
)


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    return home


@pytest.fixture
def broker(home):
    return AwayBridgeBroker(hermes_home=home, busy_ms=2000)


def _arm(broker, key="agent:main:telegram:dm:s1", surface="chat"):
    return broker.arm(key, surface=surface)


def _prepare(broker, key="agent:main:telegram:dm:s1", question="Proceed?"):
    return broker.prepare(key, question)


# ---------------------------------------------------------------------------
# arm / dedupe
# ---------------------------------------------------------------------------

def test_arm_inserts_and_reports(broker):
    result = _arm(broker)
    assert result["action"] == "inserted"
    assert broker.has_armed()


def test_arm_dedupes_same_lineage(home, broker):
    """arm → rotate key (simulate compression) → arm again → refreshed, one row."""
    _arm(broker, "sess-K0")
    # Simulate compression rotation: state.db lineage K0 → K1.
    _make_lineage(home, "sess-K0", "sess-K1")
    result = broker.arm("sess-K1")
    assert result["action"] == "refreshed"
    assert result["session_key"] == "sess-K0"
    rows = _armed_rows(broker)
    assert len(rows) == 1
    assert rows[0]["session_key"] == "sess-K0"


def test_arm_second_session_inserts_separate_row(broker):
    _arm(broker, "sess-A")
    _arm(broker, "sess-B")
    assert len(_armed_rows(broker)) == 2


# ---------------------------------------------------------------------------
# prepare: two-sided lineage-relative lookup (§4.1, R4-2)
# ---------------------------------------------------------------------------

def test_prepare_requires_armed_row(broker):
    with pytest.raises(NotArmedError):
        _prepare(broker)


def test_prepare_after_compression_two_sided(home, broker):
    """Arm at K0; both sides rotate; prepare with current key K1 still finds it."""
    _arm(broker, "sess-K0")
    _make_lineage(home, "sess-K0", "sess-K1")
    req = broker.prepare("sess-K1", "Proceed?")
    # The request inherits the ARM-TIME key (§4.1 normative).
    assert req["session_key"] == "sess-K0"
    assert req["token"].startswith("A#")
    assert len(req["token"]) == 18  # A# + 16
    row = broker.get_request(req["req_key"])
    assert row["state"] == "prepared"


def test_prepare_single_awaiting_invariant(broker, home):
    _arm(broker, "sess-K0")
    first = _prepare(broker, "sess-K0", "Q1")
    broker.publish(first["req_key"])
    with pytest.raises(AlreadyAwaitingError):
        _prepare(broker, "sess-K0", "Q2")


def test_prepare_second_session_unaffected(broker):
    _arm(broker, "sess-A")
    _arm(broker, "sess-B")
    req_a = broker.prepare("sess-A", "QA")
    req_b = broker.prepare("sess-B", "QB")
    assert req_a["session_key"] == "sess-A"
    assert req_b["session_key"] == "sess-B"


# ---------------------------------------------------------------------------
# publish / claim / first-answer-wins (§4.2) with REAL concurrent connections
# ---------------------------------------------------------------------------

def test_publish_requires_prepared(broker):
    req = {"req_key": "ab" + "0" * 32}
    with pytest.raises(Exception):
        broker.publish(req["req_key"])


def test_claim_first_answer_wins_concurrent(home):
    """Two real connections race the conditional update; exactly one wins."""
    broker_a = AwayBridgeBroker(hermes_home=home, busy_ms=2000)
    broker_b = AwayBridgeBroker(hermes_home=home, busy_ms=2000)
    broker_a.arm("sess-K0")
    req = broker_a.prepare("sess-K0", "Pick one")
    broker_a.publish(req["req_key"], waiter_pid=os.getpid(), waiter_start_time=123)

    barrier = threading.Barrier(2)
    results = {}

    def _claim(name, channel):
        b = broker_a if name == "a" else broker_b
        barrier.wait()
        results[name] = b.claim(req["req_key"], channel, "answer-%s" % name)

    t1 = threading.Thread(target=_claim, args=("a", "chat"))
    t2 = threading.Thread(target=_claim, args=("b", "telegram"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    outcomes = sorted(r["outcome"] for r in results.values())
    assert outcomes == ["lost", "won"]
    winners = [r for r in results.values() if r["outcome"] == "won"]
    loser = [r for r in results.values() if r["outcome"] == "lost"][0]
    assert loser["winner"] == winners[0]["channel"]
    row = broker_a.get_request(req["req_key"])
    assert row["state"] == "claimed"
    assert row["winning_channel"] == winners[0]["channel"]


def test_duplicate_ack_idempotent(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"])
    broker.claim(req["req_key"], "chat", "a1")
    first = broker.ack_duplicate(req["req_key"], "telegram")
    second = broker.ack_duplicate(req["req_key"], "telegram")
    assert first is True
    assert second is False


def test_claim_unknown_and_prepared_rejected(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")  # still 'prepared'
    assert broker.claim(req["req_key"], "chat", "x")["outcome"] == "reject"
    assert broker.claim("ab" + "f" * 32, "chat", "x")["outcome"] == "reject"


# ---------------------------------------------------------------------------
# waiter poll set (§4.3 complete)
# ---------------------------------------------------------------------------

def test_poll_set_all_branches(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    # prepared + armed → keep waiting
    assert broker.poll_request(req["req_key"])["outcome"] == "KEEP_WAITING"
    broker.publish(req["req_key"])
    # awaiting + armed → keep waiting
    assert broker.poll_request(req["req_key"])["outcome"] == "KEEP_WAITING"
    broker.disarm_chat("sess-K0")
    # awaiting + disarmed → DISARMED (row stays awaiting, §4.6)
    assert broker.poll_request(req["req_key"])["outcome"] == "DISARMED"
    row = broker.get_request(req["req_key"])
    assert row["state"] == "awaiting"


def test_poll_prepared_disarmed_exits(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.disarm_chat("sess-K0")
    assert broker.poll_request(req["req_key"])["outcome"] == "DISARMED"


def test_poll_claimed_maps_channel(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"])
    broker.claim(req["req_key"], "telegram", "tg")
    assert broker.poll_request(req["req_key"])["outcome"] == "TELEGRAM_WON"
    # cancelled
    broker.arm("sess-K2")
    req2 = _prepare(broker, "sess-K2")
    broker.publish(req2["req_key"])
    broker.cancel(req2["req_key"])
    assert broker.poll_request(req2["req_key"])["outcome"] == "CANCELLED"


def test_poll_no_row_is_error(broker):
    assert broker.poll_request("ab" + "e" * 32)["outcome"] == "ERROR"


def test_poll_db_error_raises(broker, home):
    # Missing parent dir → connection error path; broker must raise, waiter exits ERROR.
    broken = AwayBridgeBroker(db_path=home / "nope" / "bridge.db", busy_ms=10)
    # dir gets created by _connect; corrupt the file instead
    broken.db_path.parent.mkdir(parents=True, exist_ok=True)
    broken.db_path.write_bytes(b"this is not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        broken.poll_request("ab" + "0" * 32)


# ---------------------------------------------------------------------------
# disarm semantics (§4.6)
# ---------------------------------------------------------------------------

def test_disarm_chat_flips_every_lineage_row(home, broker):
    """arm K0 → compress K1 → re-arm refreshes (dedupe) → disarm flips all."""
    broker.arm("sess-K0")
    _make_lineage(home, "sess-K0", "sess-K1")
    broker.arm("sess-K1")  # dedupe → refresh of sess-K0
    result = broker.disarm_chat("sess-K1")
    assert result["flipped"] == ["sess-K0"]


def test_disarm_chat_leaves_other_sessions(broker):
    broker.arm("sess-A")
    broker.arm("sess-B")
    broker.disarm_chat("sess-A")
    rows = {r["session_key"]: r["status"] for r in _armed_rows(broker)}
    assert rows["sess-A"] == "disarmed"
    assert rows["sess-B"] == "armed"


def test_disarm_all_global(broker):
    broker.arm("sess-A")
    broker.arm("sess-B")
    result = broker.disarm_all()
    assert result["disarmed_count"] == 2
    assert not broker.has_armed()


# ---------------------------------------------------------------------------
# fetch idempotency (§4.3)
# ---------------------------------------------------------------------------

def test_fetch_marks_delivered_once(broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"])
    broker.claim(req["req_key"], "telegram", "the answer")
    first = broker.fetch(req["req_key"])
    second = broker.fetch(req["req_key"])
    assert first["delivered"] is True
    assert first["answer"] == "the answer"
    assert second["delivered"] is True


# ---------------------------------------------------------------------------
# sweep + GC (§4.4, §4.7)
# ---------------------------------------------------------------------------

def test_sweep_rate_limit_and_recheck(home, broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"])
    due = broker.sweep_candidates(window_hours=1.0)
    assert [c["req_key"] for c in due] == [req["req_key"]]
    # Immediately due again? No — rate-limited by last_swept_at.
    due2 = broker.sweep_candidates(window_hours=1.0)
    assert due2 == []


def test_gc_orphans_armed_rows_without_lineage_tip(home, broker):
    broker.arm("sess-GONE")
    result = broker.gc(retention_days=30)
    assert result["orphaned"] == ["sess-GONE"]
    assert not broker.has_armed()


def test_gc_keeps_armed_rows_with_open_requests(home, broker):
    broker.arm("sess-GONE")
    req = _prepare(broker, "sess-GONE")
    broker.publish(req["req_key"])
    result = broker.gc(retention_days=30)
    assert result["orphaned"] == []
    assert broker.has_armed()


def test_gc_prunes_old_delivered(home, broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"])
    broker.claim(req["req_key"], "chat", "old")
    broker.fetch(req["req_key"])
    # Backdate claimed_at beyond retention.
    with sqlite3.connect(str(broker.db_path)) as conn:
        conn.execute("UPDATE requests SET claimed_at=? WHERE req_key=?",
                     (time.time() - 40 * 86400, req["req_key"]))
        conn.commit()
    result = broker.gc(retention_days=30)
    assert result["pruned_requests"] == 1


# ---------------------------------------------------------------------------
# reconcile reporter (§4.7)
# ---------------------------------------------------------------------------

def test_reconcile_reports_all_classes(home, broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"], waiter_pid=os.getpid(), waiter_start_time=1)
    # waiter_start_time=1 will not match this process → reported dead.
    report = broker.reconcile()
    assert any(r["req_key"] == req["req_key"] for r in report["awaiting_waiter_dead"])

    broker.arm("sess-K9")
    req2 = _prepare(broker, "sess-K9")  # stuck prepared (never published)
    report = broker.reconcile()
    assert any(r["req_key"] == req2["req_key"] for r in report["prepared_stuck"])

    # Undelivered winner.
    broker.claim(req["req_key"], "telegram", "stored answer")
    report = broker.reconcile()
    assert any(r["req_key"] == req["req_key"] for r in report["undelivered_winners"])


def test_reconcile_waiter_live_identity(home, broker):
    broker.arm("sess-K0")
    req = _prepare(broker, "sess-K0")
    broker.publish(req["req_key"], waiter_pid=os.getpid(), waiter_start_time=None)
    # start_time=None → legacy liveness fallback (alive PID passes).
    report = broker.reconcile()
    assert not any(r["req_key"] == req["req_key"] for r in report["awaiting_waiter_dead"])


# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------

def test_load_config_defaults_and_overrides(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as _cfgmod

    _cfgmod._LOAD_CONFIG_CACHE.clear()
    cfg = load_away_bridge_config(home)
    assert cfg["enabled"] is False
    assert cfg["waiter_poll_seconds"] == 2
    (home / "config.yaml").write_text(
        "away_bridge:\n"
        "  enabled: true\n"
        "  waiter_poll_seconds: 5\n"
        "  telegram:\n"
        "    owner_chat_id: \"123\"\n"
        "    owner_user_id: \"123\"\n"
    )
    _cfgmod._LOAD_CONFIG_CACHE.clear()
    cfg = load_away_bridge_config(home)
    assert cfg["enabled"] is True
    assert cfg["waiter_poll_seconds"] == 5
    assert cfg["telegram"]["owner_chat_id"] == "123"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _armed_rows(broker):
    import sqlite3 as _s

    with _s.connect(str(broker.db_path)) as conn:
        conn.row_factory = _s.Row
        return [dict(r) for r in conn.execute("SELECT * FROM armed_sessions").fetchall()]


def _make_lineage(home, parent_key, child_key):
    """Insert a compression lineage parent→child in the profile state.db.

    SessionDB resolves keys through the sessions table (the gateway passes
    session keys as ids there); we create real rows so
    resolve_resume_session_id walks parent → child.
    """
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    try:
        now = time.time()
        db._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, source, started_at, end_reason, parent_session_id) "
            "VALUES (?, 'desktop', ?, NULL, NULL)",
            (parent_key, now - 60),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO messages (session_id, role, content, timestamp, active) "
            "VALUES (?, 'user', 'seed', ?, 1)",
            (parent_key, now - 50),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, source, started_at, end_reason, parent_session_id) "
            "VALUES (?, 'desktop', ?, 'compression', ?)",
            (child_key, now - 10, parent_key),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO messages (session_id, role, content, timestamp, active) "
            "VALUES (?, 'user', 'continuation', ?, 1)",
            (child_key, now - 5),
        )
        db._conn.commit()
    finally:
        try:
            db._conn.close()
        except Exception:
            pass
