"""Persistent child contracts: ownership, admission fencing, discovery and retention.

Real temporary SessionDBs exercise the transaction and transcript boundaries;
process tests catch races that an in-process lock or mocked database would hide.
"""

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools.delegate_session_store import DelegateSessionError, DelegateSessionStore


@pytest.fixture
def state(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", "cli")
    store = DelegateSessionStore(db)
    yield db, store
    db.close()


def register(db, store, sid="child", parent="parent", **kwargs):
    db.create_session(sid, "subagent", parent_session_id=parent,
                      model_config={"_delegate_from": parent})
    return store.register(parent, sid, model="chosen-model", provider="chosen-provider",
                          assignment="Review the implementation", **kwargs)


def test_completed_child_survives_reconstruction_and_close_retains_history(state):
    db, store = state
    route = {"kind": "named", "provider": "named-route"}
    register(db, store, context="Initial constraints", title="Review", workspace="/review/project", forked_context=True,
             route_descriptor=route)
    _, first = store.acquire("parent", "child", "First assignment")
    db.append_message("child", "user", "First assignment")
    db.append_message("child", "assistant", "Result one")
    assert store.release("child", first)
    path = db.db_path
    db.close()
    reopened = SessionDB(db_path=path)
    try:
        replacement = DelegateSessionStore(reopened)
        row, second = replacement.acquire("parent", "child", "Follow up", model="alternate-model")
        assert row["initial_goal"] == "Review the implementation"
        assert row["assignment"] == "Follow up"
        assert row["title"] == "Review"
        assert row["context"] == "Initial constraints"
        assert row["workspace"] == "/review/project"
        assert row["forked_context"]
        assert row["route_descriptor"] == route
        assert (row["provider"], row["model"]) == ("chosen-provider", "alternate-model")
        assert not replacement.release("child", first)  # late cleanup cannot free the new run
        with pytest.raises(DelegateSessionError, match="Stop") as busy:
            replacement.close("parent", "child")
        assert busy.value.code == "busy"
        assert replacement.release("child", second)
        replacement.close("parent", "child")
        with pytest.raises(DelegateSessionError) as closed:
            replacement.acquire("parent", "child", "Do more")
        assert closed.value.code == "closed"
        assert replacement.list("parent") == []
        assert replacement.list("parent", query="Review")[0]["status"] == "closed"
        assert replacement.observe("parent", "child")["messages"][-1]["content"] == "Result one"
    finally:
        reopened.close()


def test_ownership_follows_compression_but_never_a_fork_or_other_profile(state, tmp_path):
    db, store = state
    register(db, store)
    db.end_session("parent", "compression")
    db.create_session("parent-tip", "cli", parent_session_id="parent")
    assert store.get("parent-tip", "child")["session_id"] == "child"
    # A branch and delegated child of the same compressed parent are independent owners.
    for sid, source, marker in [("branch", "cli", "_branched_from"),
                                ("sibling", "subagent", "_delegate_from")]:
        db.create_session(sid, source, parent_session_id="parent", model_config={marker: "parent"})
        with pytest.raises(DelegateSessionError) as foreign:
            store.get(sid, "child")
        assert foreign.value.code == "unavailable"
        assert store.list(sid, query="Review") == []
    other = SessionDB(db_path=tmp_path / "other-profile.db")
    try:
        other.create_session("parent", "cli")
        with pytest.raises(DelegateSessionError):
            DelegateSessionStore(other).get("parent", "child")
    finally:
        other.close()
    # Compression changes the execution tip, never the public child ID.
    db.end_session("child", "compression")
    db.create_session("child-tip", "subagent", parent_session_id="child",
                      model_config={"_delegate_from": "parent"})
    row, token = store.acquire("parent-tip", "child", "Continue")
    assert row["session_id"] == "child"
    assert row["current_session_id"] == "child-tip"
    assert store.release("child", token, current_session_id="child-tip")


def _claim_worker(path, ready, start, results, finish):
    db = SessionDB(db_path=Path(path))
    try:
        store = DelegateSessionStore(db)
        ready.put(True)
        start.wait(20)
        try:
            _, token = store.acquire("parent", "child", "Competing assignment")
            results.put(("acquired", token))
        except DelegateSessionError as exc:
            results.put((exc.code, None))
        finish.wait(20)
    finally:
        db.close()


def test_independent_processes_cannot_both_resume_one_child(state):
    db, store = state
    register(db, store)
    ctx = multiprocessing.get_context("spawn")
    ready, results = ctx.Queue(), ctx.Queue()
    start, finish = ctx.Event(), ctx.Event()
    workers = [ctx.Process(target=_claim_worker, args=(str(db.db_path), ready, start, results, finish))
               for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=20)
        start.set()
        claims = [results.get(timeout=20) for _ in workers]
        assert sorted(code for code, _ in claims) == ["acquired", "busy"]
        winner = next(token for code, token in claims if code == "acquired")
        assert store.get("parent", "child")["status"] == "running"
        assert not store.release("child", "foreign-token")
        assert store.release("child", winner)
    finally:
        finish.set()
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        ready.close()
        results.close()
    assert all(worker.exitcode == 0 for worker in workers)


@pytest.mark.platforms("linux")
def test_dead_process_becomes_idle_without_replaying_or_expiring_live_process(state):
    db, store = state
    register(db, store)
    ctx = multiprocessing.get_context("spawn")
    ready, results = ctx.Queue(), ctx.Queue()
    start, finish = ctx.Event(), ctx.Event()
    worker = ctx.Process(target=_claim_worker, args=(str(db.db_path), ready, start, results, finish))
    try:
        worker.start()
        assert ready.get(timeout=20)
        start.set()
        code, old_token = results.get(timeout=20)
        assert code == "acquired"
        # Very old activity is not proof that a live owner has died.
        db._write_sql("UPDATE delegate_sessions SET last_activity_at = 1 WHERE session_id = 'child'")
        assert store.get("parent", "child")["status"] == "running"
        finish.set()
        worker.join(timeout=20)
        assert worker.exitcode == 0
        recovered = store.get("parent", "child")
        assert recovered["status"] == "idle"
        assert recovered["last_activity_at"] == 1  # recovery is not activity
        assert db.get_messages("child") == []  # no automatic replay of the abandoned assignment
        _, new_token = store.acquire("parent", "child", "Explicit continuation")
        assert not store.release("child", old_token)
        assert store.release("child", new_token)
    finally:
        finish.set()
        worker.join(timeout=20)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        ready.close()
        results.close()


def test_discovery_uses_activity_searches_beyond_window_and_observation_is_bounded(state, monkeypatch):
    db, store = state
    epoch = 1_700_000_000.0
    clock = SimpleNamespace(value=epoch + 100)
    monkeypatch.setattr("tools.delegate_session_store.time", SimpleNamespace(time=lambda: clock.value))
    register(db, store, "older", title="Old review")
    clock.value = epoch + 200
    register(db, store, "newer", title="New review")
    db.append_message("older", "user", "Please inspect uncommon-needle", timestamp=epoch + 300)
    db.append_message("older", "assistant", "A" * 1000, reasoning="private reasoning", timestamp=epoch + 301)
    db.append_message("older", "tool", "B" * 1000, timestamp=epoch + 302)
    clock.value = epoch + 400
    assert store.list("parent", limit=1)[0]["session_id"] == "older"
    before = store.get("parent", "older")
    result = store.observe("parent", "older", limit=2, max_chars=50)
    assert sum(len(m["content"]) for m in result["messages"]) <= 50
    assert result["truncated"]
    assert all(set(m) == {"role", "content", "timestamp"} for m in result["messages"])
    full = store.observe("parent", "older")
    assert full["messages"][0]["role"] == "user"
    assert full["messages"][0]["timestamp"] == epoch + 300
    assert store.get("parent", "older")["last_activity_at"] == before["last_activity_at"] == epoch + 302
    _, token = store.acquire("parent", "newer", "Run again")
    assert store.list("parent", limit=1)[0]["session_id"] == "newer"
    assert store.list("parent", query="uncommon-needle", limit=1)[0]["session_id"] == "older"
    assert store.get("parent", "older")["created_at"] == epoch + 100
    assert store.release("newer", token)


def test_deleted_transcript_is_unavailable_and_never_recreated(state):
    db, store = state
    register(db, store)
    db.delete_session("child")
    with pytest.raises(DelegateSessionError) as deleted:
        store.acquire("parent", "child", "Resume")
    assert deleted.value.code == "unavailable"
    assert store.list("parent") == []
    assert db.get_session("child") is None
    register(db, store, "rotated")
    db.end_session("rotated", "compression")
    db.create_session("tip", "subagent", parent_session_id="rotated")
    assert store.get("parent", "rotated")["current_session_id"] == "tip"
    db.delete_session("tip")
    with pytest.raises(DelegateSessionError):
        store.acquire("parent", "rotated", "Do not fall back to stale history")


def test_older_registry_adds_optional_fields_without_losing_conversation(state):
    db, store = state
    register(db, store)
    for column in ("workspace", "forked_context", "route_descriptor"):
        db._write_sql(f"ALTER TABLE delegate_sessions DROP COLUMN {column}")
    updated = DelegateSessionStore(db)
    row, token = updated.acquire("parent", "child", "Continue after upgrade")
    assert row["initial_goal"] == "Review the implementation"
    assert row["workspace"] is None
    assert row["forked_context"] is False
    assert row["route_descriptor"] is None
    assert updated.release("child", token)


def test_route_descriptor_cannot_persist_a_credential_bundle(state):
    db, store = state
    with pytest.raises(DelegateSessionError) as rejected:
        register(db, store, route_descriptor={"kind": "named", "api_key": "must-not-be-stored"})
    assert rejected.value.code == "invalid_route"
    assert store.list("parent") == []


def test_query_does_not_block_foreground_transcript_writes(state, monkeypatch):
    """A paused transcript search must not own SessionDB's writer lock (P3-2)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    db, store = state
    register(db, store)
    db.append_message('child', 'assistant', 'searchable finding')
    scanning, finish = threading.Event(), threading.Event()
    matches = store._matches

    def held_scan(conn, row, needle):
        scanning.set()
        assert finish.wait(20)
        return matches(conn, row, needle)

    monkeypatch.setattr(store, '_matches', held_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        query = pool.submit(store.list, 'parent', query='searchable')
        try:
            assert scanning.wait(10)
            write = pool.submit(db.append_message, 'parent', 'user', 'Foreground turn')
            write.result(timeout=10)
        finally:
            finish.set()
        assert query.result(timeout=10)[0]['session_id'] == 'child'
    assert db.get_messages('parent')[-1]['content'] == 'Foreground turn'


def test_abandonment_retirement_is_token_fenced_and_cannot_be_reopened(state):
    db, store = state
    register(db, store)
    _, old = store.acquire('parent', 'child', 'First')
    assert store.release('child', old)
    _, current = store.acquire('parent', 'child', 'Next')
    assert not store.retire_abandoned('child', old)
    assert store.get('parent', 'child')['status'] == 'running'
    assert store.retire_abandoned('child', current)
    assert not store.release('child', current)
    assert store.close('parent', 'child')['status'] == 'closed'
    with pytest.raises(DelegateSessionError) as refused:
        store.acquire('parent', 'child', 'Unsafe overlapping worker')
    assert refused.value.code == 'closed'
