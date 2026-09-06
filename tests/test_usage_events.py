"""Behavior contracts for profile-local request events (no network)."""
from hermes_state import SessionDB


def test_timeline_exact_boundaries_unknowns_and_provider_filter(tmp_path, monkeypatch):
    from hermes_state_usage_events import UsageEvent
    import hermes_state_usage_events as events
    end = 2_000_000_123_456_789
    width = 900_000_000
    start = end - 24 * width
    with SessionDB(tmp_path / "state.db") as db:
        for i, when in enumerate([start - 1, start, start + width - 1,
                                  start + width, end - 1, end, end + 1]):
            assert db.record_usage_event(UsageEvent(str(i), "openai-codex", "test", "p", when,
                                                  input_tokens=10, output_tokens=3,
                                                  cache_read_tokens=7, reasoning_tokens=2), now_us=end)
        for id_, provider, tokens in [("other", "anthropic", 999), ("missing", "openai-codex", None),
                                     ("zero", "openai-codex", 0)]:
            assert db.record_usage_event(UsageEvent(id_, provider, "test", "p", start,
                                                  input_tokens=tokens, output_tokens=tokens), now_us=end)
        assert hasattr(db, "codex_usage_timeline")
        monkeypatch.setattr(events.time, "time_ns", lambda: end * 1000)
        result = db.codex_usage_timeline()
        assert result["as_of_us"] == end
        bins = result["bins"]
        assert len(bins) == 24
        assert bins[0]["start_us"] == start and bins[-1]["end_us"] == end
        assert all(a["end_us"] == b["start_us"] for a, b in zip(bins, bins[1:]))
        assert all(b["end_us"] - b["start_us"] == width for b in bins)
        total = result["total"]
        assert total["processed_tokens"] == 52
        assert total["events"] == 6
        assert total["unknown_input_tokens"] == 1
        assert total["unknown_output_tokens"] == 1
        independent = db._read_one("SELECT SUM(input_tokens + output_tokens) FROM usage_events "
                                  "WHERE provider=? AND completed_at_us>=? AND completed_at_us<?",
                                  ("openai-codex", start, end))[0]
        assert total["processed_tokens"] == independent
        for field in total:
            assert sum(b["usage"][field] for b in bins) == total[field]
        assert bins[0]["usage"]["processed_tokens"] == 26
        assert bins[1]["usage"]["processed_tokens"] == 13
        assert result["coverage"]["status"] == "partial"


def test_upgrade_is_additive_and_duplicate_attempt_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("existing", source="cli")
        # Reproduce a pre-event store without pinning the current schema version.
        db._write_sql("DROP TABLE IF EXISTS usage_events")
    with SessionDB(path) as db:
        assert db.get_session("existing") is not None
        assert hasattr(db, "record_usage_event")
        from hermes_state_usage_events import UsageEvent
        event = UsageEvent(attempt_id="attempt-a", provider="openai-codex", model="test-model",
                           profile="test-profile", completed_at_us=1, input_tokens=10, output_tokens=2)
        assert db.record_usage_event(event, now_us=1)
        assert db.record_usage_event(event, now_us=1)
        rows = db._read_all("SELECT * FROM usage_events")
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 10
    with SessionDB(path) as db:
        assert len(db._read_all("SELECT * FROM usage_events")) == 1
        assert db.get_session("existing") is not None



def test_retention_is_bounded_on_ingestion_and_never_runs_on_reads(tmp_path, monkeypatch):
    import hermes_state_usage_events as events
    from hermes_state_usage_events import UsageEvent
    now = 2_000_000_000_000_000
    cutoff = now - 7 * 24 * 3600 * 1_000_000
    with SessionDB(tmp_path / "state.db") as db:
        for i in range(300):
            assert db.record_usage_event(UsageEvent(str(i), "openai-codex", "test", "p", cutoff - 1),
                                         now_us=cutoff)
        monkeypatch.setattr(events.time, "time_ns", lambda: now * 1000)
        db.codex_usage_timeline()
        assert db._read_one("SELECT COUNT(*) FROM usage_events")[0] == 300
        assert db.record_usage_event(UsageEvent("edge", "openai-codex", "test", "p", cutoff), now_us=now)
        remaining = db._read_one("SELECT COUNT(*) FROM usage_events WHERE completed_at_us<?", (cutoff,))[0]
        assert 0 < remaining < 300
        assert hasattr(db, "prune_usage_events")
        assert db.prune_usage_events(now_us=now) == remaining
        assert db._read_one("SELECT COUNT(*) FROM usage_events")[0] == 1
        # Expired duplicate delivery cannot resurrect an evicted observation.
        assert db.record_usage_event(UsageEvent("0", "openai-codex", "test", "p", cutoff - 1), now_us=now)
        assert db._read_one("SELECT COUNT(*) FROM usage_events")[0] == 1


def test_lock_failure_is_detectable_and_recovery_does_not_lose_marker(tmp_path, monkeypatch, caplog):
    import sqlite3
    import time
    from hermes_state_usage_events import UsageEvent
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        event = UsageEvent("locked", "openai-codex", "test", "p", time.time_ns() // 1000,
                           input_tokens=0, output_tokens=0)
        blocker = sqlite3.connect(path, isolation_level=None)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            assert db.record_usage_event(event) is False
            assert time.monotonic() - started < 3  # SQLite's 1s busy handler + bounded app patience.
            assert db.codex_usage_timeline()["coverage"]["recording_failures_in_process"] >= 1
            assert "usage_event_recording_failed" in caplog.text
            assert "locked" not in caplog.text
        finally:
            blocker.rollback()
            blocker.close()
        assert db.record_usage_event(event)
    with SessionDB(path, read_only=True) as db:
        coverage = db.codex_usage_timeline()["coverage"]
        assert coverage["persisted_recording_failure"] is True
        assert db.record_usage_event(event) is False


def test_query_failure_does_not_return_complete_looking_zero(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db._write_sql("DROP TABLE usage_events")
        result = db.codex_usage_timeline()
        assert result["coverage"]["status"] == "unavailable"
        assert result["total"] is None
        assert all(b["usage"] is None for b in result["bins"])



def _concurrent_writer(path, worker, ready, start, results):
    from hermes_state_usage_events import UsageEvent
    import time
    with SessionDB(path) as db:
        ready.put(worker)
        assert start.wait(30)
        ok = True
        for i in range(30):
            # All processes observe one shared attempt plus their own distinct attempts.
            for id_ in ("shared", f"{worker}-{i}"):
                ok = db.record_usage_event(UsageEvent(id_, "openai-codex", "test", "p",
                                                       time.time_ns() // 1000, input_tokens=2,
                                                       output_tokens=1)) and ok
        results.put(ok)


def test_multiprocess_writers_converge_and_profiles_remain_isolated(tmp_path):
    import multiprocessing
    path = tmp_path / "a" / "state.db"
    with SessionDB(path):
        pass
    ctx = multiprocessing.get_context("spawn")
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_concurrent_writer, args=(path, i, ready, start, results)) for i in range(3)]
    try:
        for p in workers:
            p.start()
        assert {ready.get(timeout=45) for _ in workers} == {0, 1, 2}
        start.set()
        assert all([results.get(timeout=45) for _ in workers])
        for p in workers:
            p.join(30)
            assert p.exitcode == 0
        with SessionDB(path) as db, SessionDB(tmp_path / "b" / "state.db") as other:
            result = db.codex_usage_timeline()
            assert result["total"]["events"] == 91
            assert result["total"]["processed_tokens"] == 273
            assert other.codex_usage_timeline()["total"]["events"] == 0
            assert db._read_one("PRAGMA integrity_check")[0] == "ok"
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()
            p.join(10)
        ready.close()
        results.close()
