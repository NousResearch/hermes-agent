"""Canonical state.db bootstrap coverage for async delegation startup."""

import hashlib
import json
import multiprocessing
import os
import queue
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_state
from hermes_state import SCHEMA_VERSION, SessionDB
from tools import async_delegation as ad
from tools.process_registry import ProcessRegistry


_CANONICAL_TABLES = {
    "schema_version",
    "sessions",
    "messages",
    "session_model_usage",
    "state_meta",
    "gateway_routing",
    "compression_locks",
    "messages_fts",
}
_COMPLETE_TABLES = _CANONICAL_TABLES | {"async_delegations"}
_ASYNC_COLUMNS = {
    "owner_pid",
    "owner_started_at",
    "task_json",
    "delivery_claim",
    "delivery_claimed_at",
}
_BASE_FTS_TRIGGERS = {
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
}
_TRIGRAM_FTS_TRIGGERS = {
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
}

_ASYNC_ONLY_SCHEMA = """
CREATE TABLE async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at REAL
);
"""


def _table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


def _assert_complete_schema(db_path):
    assert _COMPLETE_TABLES <= _table_names(db_path)
    assert _ASYNC_COLUMNS <= _column_names(db_path, "async_delegations")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    assert row == (SCHEMA_VERSION,)


def _column_names(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        return {
            row[1]
            for row in conn.execute(f'PRAGMA table_info("{table_name}")')
        }


def _trigger_names(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }


def _sqlite_master_sql(db_path, object_name):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (object_name,)
        ).fetchone()
    return None if row is None else row[0]


@pytest.fixture(autouse=True)
def _reset_async_delegation_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _process_schema_worker(kind, home, start_event, ready_queue, result_queue):
    """Run one production startup path in a fresh spawned interpreter."""
    os.environ["HERMES_HOME"] = str(home)
    ready_queue.put({"worker": kind, "pid": os.getpid()})
    try:
        if not start_event.wait(timeout=15):
            raise TimeoutError("timed out waiting for synchronized process start")
        if kind == "session":
            db = SessionDB(db_path=home / "state.db")
            db.close()
            result = None
        elif kind == "async":
            result = ad.restore_undelivered_completions(queue.Queue())
        else:  # pragma: no cover - test harness guard
            raise ValueError(f"unknown worker kind: {kind}")
        result_queue.put({"worker": kind, "ok": True, "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "worker": kind,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_process_schema_case(ctx, home, worker_kinds):
    start_event = ctx.Event()
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_process_schema_worker,
            args=(kind, home, start_event, ready_queue, result_queue),
        )
        for kind in worker_kinds
    ]
    try:
        for process in processes:
            process.start()
        ready = [ready_queue.get(timeout=20) for _ in processes]
        start_event.set()
        for process in processes:
            process.join(timeout=30)

        outcomes = []
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                outcomes.append(
                    {
                        "worker": "unknown",
                        "ok": False,
                        "error": f"child {process.pid} timed out",
                        "traceback": "",
                    }
                )
        while len(outcomes) < len(processes):
            try:
                outcomes.append(result_queue.get(timeout=2))
            except queue.Empty:
                break
        if len(outcomes) < len(processes):
            outcomes.append(
                {
                    "worker": "unknown",
                    "ok": False,
                    "error": "child exited without publishing a startup result",
                    "traceback": "",
                }
            )
        return {
            "ready": ready,
            "exitcodes": [process.exitcode for process in processes],
            "outcomes": outcomes,
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for process_queue in (ready_queue, result_queue):
            process_queue.close()
            process_queue.join_thread()


def _prepare_async_only_db(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_ASYNC_ONLY_SCHEMA)
        conn.execute(
            """INSERT INTO async_delegations (
                   delegation_id, origin_session, state, dispatched_at,
                   updated_at, delivery_state
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("deleg_existing", "origin", "completed", 1.0, 2.0, "delivered"),
        )


def _assert_existing_delegation_preserved(db_path):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT origin_session, state, delivery_state "
            "FROM async_delegations WHERE delegation_id = ?",
            ("deleg_existing",),
        ).fetchone()
    assert row == ("origin", "completed", "delivered")


def _patch_async_connect_with_journal_failure(monkeypatch, reason):
    real_connect = sqlite3.connect
    opened = []

    class _TrackedConnection(sqlite3.Connection):
        tracker: dict

        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace(" ", "")
            if "journal_mode=wal" in normalized:
                self.tracker["wal_attempts"] += 1
                raise sqlite3.OperationalError(reason)
            if "journal_mode=delete" in normalized:
                self.tracker["delete_attempts"] += 1
            return super().execute(sql, *args, **kwargs)

        def close(self):  # type: ignore[override]
            self.tracker["closed"] = True
            return super().close()

    def tracked_connect(path, *args, **kwargs):
        tracker = {"wal_attempts": 0, "delete_attempts": 0, "closed": False}
        kwargs["factory"] = _TrackedConnection
        conn = real_connect(path, *args, **kwargs)
        setattr(conn, "tracker", tracker)
        opened.append((conn, tracker))
        return conn

    monkeypatch.setattr(ad.sqlite3, "connect", tracked_connect)
    return opened


def _schema_transaction_holder(db_path, ready_event, release_event, crash):
    conn = sqlite3.connect(db_path, timeout=0.1, isolation_level=None)
    try:
        hermes_state._begin_schema_transaction(conn)
        ready_event.set()
        if not release_event.wait(timeout=15):
            raise TimeoutError("schema-transaction test holder timed out")
        if crash:
            os._exit(0)
        conn.rollback()
        hermes_state._end_schema_transaction(conn)
    finally:
        conn.close()


def test_process_registry_bootstrap_initializes_canonical_schema_when_db_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    assert not db_path.exists()

    ProcessRegistry()

    _assert_complete_schema(db_path)


def test_process_registry_bootstrap_initializes_canonical_schema_for_zero_byte_db(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    db_path.touch()
    assert db_path.stat().st_size == 0

    ProcessRegistry()

    _assert_complete_schema(db_path)


def test_process_registry_bootstrap_upgrades_async_only_db_without_losing_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_ASYNC_ONLY_SCHEMA)
        conn.execute(
            """INSERT INTO async_delegations (
                   delegation_id, origin_session, state, dispatched_at,
                   updated_at, delivery_state
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("deleg_existing", "origin", "completed", 1.0, 2.0, "delivered"),
        )

    ProcessRegistry()

    _assert_complete_schema(db_path)
    assert {
        "owner_pid",
        "owner_started_at",
        "task_json",
        "delivery_claim",
        "delivery_claimed_at",
    } <= _column_names(db_path, "async_delegations")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT origin_session, state, delivery_state "
            "FROM async_delegations WHERE delegation_id = ?",
            ("deleg_existing",),
        ).fetchone()
    assert row == ("origin", "completed", "delivered")


def test_concurrent_session_and_async_first_open_produces_complete_schema(
    tmp_path, monkeypatch
):
    for attempt in range(10):
        home = tmp_path / f"attempt-{attempt}"
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        start = threading.Barrier(2)

        def open_session_db():
            start.wait(timeout=5)
            db = SessionDB(db_path=db_path)
            db.close()

        def restore_async_delegations():
            start.wait(timeout=5)
            assert ad.restore_undelivered_completions(queue.Queue()) == 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(open_session_db),
                executor.submit(restore_async_delegations),
            ]
            for future in futures:
                future.result(timeout=15)

        _assert_complete_schema(db_path)


def test_session_crud_works_after_async_first_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"

    assert ad.restore_undelivered_completions(queue.Queue()) == 0

    db = SessionDB(db_path=db_path)
    try:
        db.create_session("after-async", source="cli", model="test-model")
        message_id = db.append_message(
            "after-async", role="user", content="persisted after async startup"
        )

        assert message_id > 0
        session = db.get_session("after-async")
        assert session is not None
        assert session["model"] == "test-model"
        messages = db.get_messages("after-async")
        assert [message["content"] for message in messages] == [
            "persisted after async startup"
        ]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("initial_state", "worker_kinds"),
    [
        ("missing", ("session", "async")),
        ("missing", ("async", "async")),
        ("async-only", ("session", "async")),
        ("async-only", ("async", "async")),
    ],
    ids=[
        "missing-session-async",
        "missing-async-async",
        "async-only-session-async",
        "async-only-async-async",
    ],
)
def test_separate_process_startup_matrix(
    tmp_path, initial_state, worker_kinds
):
    """Every child must start cleanly and leave one complete shared schema.

    CI uses one iteration for a quick regression. Acceptance stress sets
    ``HERMES_SCHEMA_PROCESS_ITERATIONS=100`` for 400 synchronized fresh-home
    cases (800 separately spawned child interpreters).
    """
    iterations = int(os.environ.get("HERMES_SCHEMA_PROCESS_ITERATIONS", "1"))
    assert iterations > 0
    ctx = multiprocessing.get_context("spawn")
    failures = []

    for attempt in range(iterations):
        home = tmp_path / f"attempt-{attempt}"
        db_path = home / "state.db"
        if initial_state == "async-only":
            _prepare_async_only_db(db_path)

        result = _run_process_schema_case(ctx, home, worker_kinds)
        child_errors = [
            outcome for outcome in result["outcomes"] if not outcome.get("ok")
        ]
        abnormal_exits = [
            exitcode for exitcode in result["exitcodes"] if exitcode != 0
        ]
        try:
            _assert_complete_schema(db_path)
            if initial_state == "async-only":
                _assert_existing_delegation_preserved(db_path)
        except BaseException:
            child_errors.append(
                {
                    "worker": "schema-inspection",
                    "ok": False,
                    "error": "final schema or preserved-row assertion failed",
                    "traceback": traceback.format_exc(),
                }
            )
        if child_errors or abnormal_exits:
            failures.append(
                {
                    "attempt": attempt,
                    "initial_state": initial_state,
                    "worker_kinds": worker_kinds,
                    "abnormal_exits": abnormal_exits,
                    "result": result,
                }
            )

    assert not failures, json.dumps(failures, indent=2, sort_keys=True)


@pytest.mark.parametrize("replacement_mode", ["unlink", "atomic-replace"])
def test_same_path_db_replacement_reestablishes_canonical_schema(
    tmp_path, monkeypatch, replacement_mode
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"

    with ad._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    _assert_complete_schema(db_path)

    if replacement_mode == "unlink":
        db_path.unlink()
        _prepare_async_only_db(db_path)
    else:
        replacement = tmp_path / "replacement.db"
        _prepare_async_only_db(replacement)
        os.replace(replacement, db_path)

    assert ad.restore_undelivered_completions(queue.Queue()) == 0
    _assert_complete_schema(db_path)
    _assert_existing_delegation_preserved(db_path)


def test_replacement_between_open_and_readiness_reopens_current_object(
    tmp_path, monkeypatch
):
    """A connection opened before atomic replacement must never be returned."""
    iterations = int(os.environ.get("HERMES_SCHEMA_REPLACEMENT_ITERATIONS", "1"))
    assert iterations > 0
    production_open = ad._open_with_wal

    for attempt in range(iterations):
        home = tmp_path / f"attempt-{attempt}"
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        initial = SessionDB(db_path=db_path)
        initial.close()
        stale_observer = sqlite3.connect(db_path)

        replacement = home / "replacement.db"
        _prepare_async_only_db(replacement)
        opened = threading.Event()
        resume = threading.Event()
        first_open = True
        outcome = {}

        def paused_open(path):
            nonlocal first_open
            conn = production_open(path)
            if first_open:
                first_open = False
                opened.set()
                assert resume.wait(timeout=15)
            return conn

        def async_writer():
            try:
                with ad._connect() as conn:
                    conn.execute(
                        """INSERT INTO async_delegations
                           (delegation_id, origin_session, state, dispatched_at,
                            updated_at, delivery_state)
                           VALUES ('deleg_current', 'origin', 'completed', 1, 2,
                                   'delivered')"""
                    )
                    conn.commit()
                outcome["ok"] = True
            except BaseException:
                outcome["traceback"] = traceback.format_exc()

        monkeypatch.setattr(ad, "_open_with_wal", paused_open)
        worker = threading.Thread(target=async_writer)
        worker.start()
        assert opened.wait(timeout=15)
        os.replace(replacement, db_path)
        resume.set()
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert outcome == {"ok": True}, outcome.get("traceback")

        _assert_complete_schema(db_path)
        _assert_existing_delegation_preserved(db_path)
        with sqlite3.connect(db_path) as current:
            assert current.execute(
                "SELECT COUNT(*) FROM async_delegations "
                "WHERE delegation_id = 'deleg_current'"
            ).fetchone() == (1,)
        assert stale_observer.execute(
            "SELECT COUNT(*) FROM async_delegations "
            "WHERE delegation_id = 'deleg_current'"
        ).fetchone() == (0,)
        stale_observer.close()


@pytest.mark.parametrize(
    "damage",
    ["missing-table", "missing-trigger", "wrong-table"],
)
def test_current_version_supported_fts_damage_is_repaired(
    tmp_path, monkeypatch, damage
):
    iterations = int(os.environ.get("HERMES_SCHEMA_FTS_REPAIR_ITERATIONS", "1"))
    assert iterations > 0

    for attempt in range(iterations):
        home = tmp_path / damage / f"attempt-{attempt}"
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        db = SessionDB(db_path=db_path)
        if not db._fts_enabled:
            db.close()
            pytest.skip("SQLite runtime has no base FTS5 support")
        db.close()

        with sqlite3.connect(db_path) as conn:
            if damage in {"missing-table", "wrong-table"}:
                for trigger in _BASE_FTS_TRIGGERS:
                    conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                conn.execute("DROP TABLE messages_fts")
            if damage == "missing-trigger":
                conn.execute("DROP TRIGGER messages_fts_update")
            elif damage == "wrong-table":
                conn.execute("CREATE TABLE messages_fts(content TEXT)")
            assert conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone() == (SCHEMA_VERSION,)

        with ad._connect() as conn:
            assert conn.execute("SELECT * FROM messages_fts LIMIT 0").fetchall() == []

        master_sql = (_sqlite_master_sql(db_path, "messages_fts") or "").lower()
        assert "create virtual table" in master_sql
        assert "using fts5" in master_sql
        assert _BASE_FTS_TRIGGERS <= _trigger_names(db_path)


def test_readiness_does_not_require_optional_trigram_when_tokenizer_is_missing(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    real_connect = sqlite3.connect

    class _NoTrigramCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):  # type: ignore[override]
            if "tokenize='trigram'" in sql:
                raise sqlite3.OperationalError("no such tokenizer: trigram")
            return super().execute(sql, parameters)

    class _NoTrigramConnection(sqlite3.Connection):
        def cursor(self, factory=None):
            return super().cursor(factory or _NoTrigramCursor)

        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace(" ", "")
            if "temp._hermes_fts_trigram_probeusingfts5" in normalized:
                raise sqlite3.OperationalError("no such tokenizer: trigram")
            return super().execute(sql, *args, **kwargs)

    def connect_without_trigram(*args, **kwargs):
        kwargs["factory"] = _NoTrigramConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", connect_without_trigram)
    db = SessionDB(db_path=db_path)
    db.close()

    conn = real_connect(db_path, factory=_NoTrigramConnection)
    try:
        assert ad._canonical_schema_ready(conn, SCHEMA_VERSION)
    finally:
        conn.close()


@pytest.mark.parametrize("reason", ["locking protocol", "not authorized"])
def test_async_wal_open_uses_shared_delete_fallback(tmp_path, monkeypatch, reason):
    opened = _patch_async_connect_with_journal_failure(monkeypatch, reason)

    conn = ad._open_with_wal(tmp_path / "network-fs.db")
    try:
        assert opened[0][1] == {
            "wal_attempts": 1,
            "delete_attempts": 1,
            "closed": False,
        }
        conn.execute("CREATE TABLE usable_after_fallback (id INTEGER)")
    finally:
        conn.close()


def test_async_wal_open_does_not_downgrade_existing_wal(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "already-wal.db"
    with sqlite3.connect(db_path, isolation_level=None) as primer:
        assert primer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        primer.execute("CREATE TABLE preserved (id INTEGER)")

    opened = _patch_async_connect_with_journal_failure(
        monkeypatch, "locking protocol"
    )
    conn = ad._open_with_wal(db_path)
    conn.close()

    assert opened[0][1]["wal_attempts"] == 0
    assert opened[0][1]["delete_attempts"] == 0
    with sqlite3.connect(db_path) as check:
        assert check.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_async_wal_open_propagates_unrelated_error_and_closes_connection(
    tmp_path, monkeypatch
):
    opened = _patch_async_connect_with_journal_failure(
        monkeypatch, "no such table: unrelated"
    )

    with pytest.raises(sqlite3.OperationalError, match="no such table: unrelated"):
        ad._open_with_wal(tmp_path / "broken.db")

    assert opened[0][1] == {
        "wal_attempts": 1,
        "delete_attempts": 0,
        "closed": True,
    }


@pytest.mark.parametrize(
    "error_message",
    [
        "disk I/O error during ALTER",
        "disk I/O error after duplicate column name: owner_pid",
        "duplicate column name: another_column",
    ],
)
def test_canonical_reconcile_propagates_non_exact_duplicate_errors(
    tmp_path, monkeypatch, error_message
):
    db_path = tmp_path / "state.db"
    _prepare_async_only_db(db_path)
    real_connect = sqlite3.connect

    class _AlterFailsCursor(sqlite3.Cursor):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace('"', "")
            if "alter table async_delegations add column owner_pid" in normalized:
                raise sqlite3.OperationalError(error_message)
            return super().execute(sql, *args, **kwargs)

    class _AlterFailsConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):  # type: ignore[override]
            kwargs["factory"] = _AlterFailsCursor
            return super().cursor(*args, **kwargs)

    def gated_connect(database, *args, **kwargs):
        if str(database) == str(db_path):
            kwargs["factory"] = _AlterFailsConnection
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", gated_connect)
    with pytest.raises(sqlite3.OperationalError) as exc_info:
        SessionDB(db_path=db_path)
    assert str(exc_info.value) == error_message


@pytest.mark.parametrize(
    ("message", "columns", "accepted"),
    [
        ("duplicate column name: owner_pid", {"owner_pid"}, True),
        ("duplicate column name: owner_pid", set(), False),
        ("duplicate column name: another_column", {"owner_pid"}, False),
        ("disk I/O error after duplicate column name: owner_pid", {"owner_pid"}, False),
        ("DUPLICATE COLUMN NAME: owner_pid", {"owner_pid"}, False),
    ],
)
def test_duplicate_column_race_requires_exact_target_and_fresh_pragma(
    message, columns, accepted
):
    conn = sqlite3.connect(":memory:")
    try:
        declared = ", ".join(f'"{name}" INTEGER' for name in sorted(columns))
        conn.execute(f"CREATE TABLE async_delegations ({declared or 'id INTEGER'})")
        assert hermes_state._is_exact_duplicate_column_race(
            conn.cursor(),
            "async_delegations",
            "owner_pid",
            sqlite3.OperationalError(message),
        ) is accepted
    finally:
        conn.close()


def test_sqlite_serializer_ignores_unsupported_adjacent_flock(
    tmp_path, monkeypatch
):
    import errno
    import fcntl

    def unsupported(*_args, **_kwargs):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    db_path = tmp_path / "shared" / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    _assert_complete_schema(db_path)
    assert not db_path.with_name(f"{db_path.name}.schema.lock").exists()


def test_symlink_and_normalized_paths_use_one_canonical_database(tmp_path):
    target = tmp_path / "real" / "state.db"
    target.parent.mkdir()
    alias = tmp_path / "state-link.db"
    alias.symlink_to(target)

    db = SessionDB(db_path=alias)
    try:
        assert db.db_path == target.resolve()
    finally:
        db.close()
    _assert_complete_schema(target)
    assert not alias.with_name(f"{alias.name}.schema.lock").exists()


def test_hardlink_alias_is_rejected_before_mutation(tmp_path):
    iterations = int(os.environ.get("HERMES_SCHEMA_ALIAS_ITERATIONS", "1"))
    assert iterations > 0

    for attempt in range(iterations):
        home = tmp_path / f"attempt-{attempt}"
        primary = home / "state.db"
        alias = home / "alternate.db"
        db = SessionDB(db_path=primary)
        db.close()
        try:
            os.link(primary, alias)
        except OSError as exc:
            pytest.skip(f"hard links unavailable: {exc}")

        before = hashlib.sha256(primary.read_bytes()).hexdigest()
        with pytest.raises(hermes_state.StateDBAliasError, match="hard-link alias"):
            SessionDB(db_path=alias)
        after = hashlib.sha256(primary.read_bytes()).hexdigest()
        assert after == before
        assert not alias.with_name(f"{alias.name}-wal").exists()
        assert not alias.with_name(f"{alias.name}-journal").exists()
        assert os.path.samefile(primary, alias)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork required")
@pytest.mark.parametrize("child_kind", ["session", "async"])
def test_posix_fork_child_waits_for_parent_schema_transaction(
    tmp_path, monkeypatch, child_kind
):
    iterations = int(os.environ.get("HERMES_SCHEMA_FORK_ITERATIONS", "1"))
    assert iterations > 0
    ctx = multiprocessing.get_context("fork")
    for attempt in range(iterations):
        home = tmp_path / child_kind / f"attempt-{attempt}"
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        initial = SessionDB(db_path=db_path)
        initial.close()
        holder = sqlite3.connect(db_path, timeout=0.1, isolation_level=None)
        hermes_state._begin_schema_transaction(holder)
        started = ctx.Event()
        entered = ctx.Event()

        def fork_child():
            started.set()
            if child_kind == "session":
                db = SessionDB(db_path=db_path)
                db.close()
            else:
                ad.restore_undelivered_completions(queue.Queue())
            entered.set()

        child = ctx.Process(target=fork_child)
        child.start()
        try:
            assert started.wait(timeout=10)
            assert not entered.wait(timeout=0.05), (
                "fork child entered schema initialization before parent release"
            )
            holder.rollback()
            hermes_state._end_schema_transaction(holder)
            assert entered.wait(timeout=15)
        finally:
            if holder.in_transaction:
                holder.rollback()
            hermes_state._end_schema_transaction(holder)
            holder.close()
            child.join(timeout=15)
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        assert child.exitcode == 0


def test_schema_transaction_wait_is_bounded(tmp_path, monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    ready_event = ctx.Event()
    release_event = ctx.Event()
    db_path = tmp_path / "state.db"
    holder = ctx.Process(
        target=_schema_transaction_holder,
        args=(db_path, ready_event, release_event, False),
    )
    holder.start()
    try:
        assert ready_event.wait(timeout=15)
        monkeypatch.setattr(
            hermes_state, "_STATE_SCHEMA_TRANSACTION_TIMEOUT_S", 0.2
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="state schema transaction"):
            SessionDB(db_path=db_path)
        assert time.monotonic() - started < 2
    finally:
        release_event.set()
        holder.join(timeout=15)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_schema_transaction_recovers_after_owner_process_exits(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready_event = ctx.Event()
    release_event = ctx.Event()
    db_path = tmp_path / "state.db"
    holder = ctx.Process(
        target=_schema_transaction_holder,
        args=(db_path, ready_event, release_event, True),
    )
    holder.start()
    assert ready_event.wait(timeout=15)
    release_event.set()
    holder.join(timeout=15)
    assert holder.exitcode == 0

    db = SessionDB(db_path=db_path)
    db.close()
    _assert_complete_schema(db_path)
    assert not db_path.with_name(f"{db_path.name}.schema.lock").exists()

@pytest.mark.parametrize("bootstrap_kind", ["session", "async"])
def test_bootstrap_keeps_periodic_checkpoint_passive(
    tmp_path, monkeypatch, bootstrap_kind
):
    """Canonical and async-first bootstrap must retain periodic PASSIVE mode."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    real_connect = sqlite3.connect
    checkpoint_calls = []

    class _CheckpointTracingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace(" ", "")
            if "wal_checkpoint(" in normalized:
                checkpoint_calls.append((normalized, self.in_transaction))
            return super().execute(sql, *args, **kwargs)

    def tracing_connect(*args, **kwargs):
        kwargs["factory"] = _CheckpointTracingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracing_connect)
    if bootstrap_kind == "async":
        conn = ad._connect()
        conn.close()

    db = SessionDB(db_path=db_path)
    start = len(checkpoint_calls)
    try:
        for index in range(db._CHECKPOINT_EVERY_N_WRITES):
            db._execute_write(
                lambda conn, current=index: conn.execute(
                    "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                    (f"checkpoint-{current}", "test", float(current)),
                )
            )

        periodic_calls = checkpoint_calls[start:]
        assert periodic_calls == [("pragmawal_checkpoint(passive)", False)]
    finally:
        db.close()

    assert checkpoint_calls[-1] == ("pragmawal_checkpoint(truncate)", False)
    with real_connect(db_path) as check:
        assert check.execute("PRAGMA quick_check").fetchone() == ("ok",)


def test_failed_passive_checkpoint_preserves_schema_and_ownership(
    tmp_path, monkeypatch, caplog
):
    """A periodic checkpoint failure cannot partially mutate canonical schema."""
    db_path = tmp_path / "state.db"
    real_connect = sqlite3.connect
    inject_failure = False

    class _CheckpointFailureConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace(" ", "")
            if inject_failure and "wal_checkpoint(passive)" in normalized:
                raise sqlite3.OperationalError("injected PASSIVE checkpoint failure")
            return super().execute(sql, *args, **kwargs)

    def failure_connect(*args, **kwargs):
        kwargs["factory"] = _CheckpointFailureConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", failure_connect)
    db = SessionDB(db_path=db_path)
    db.create_session("checkpoint-owner", source="test")
    inject_failure = True
    try:
        with caplog.at_level("WARNING"):
            db._try_wal_checkpoint()
        assert "injected PASSIVE checkpoint failure" in caplog.text
        assert hermes_state._STATE_SCHEMA_ACTIVE_CONNECTIONS == {}
    finally:
        inject_failure = False
        db.close()

    _assert_complete_schema(db_path)
    with real_connect(db_path) as check:
        assert check.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert check.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = 'checkpoint-owner'"
        ).fetchone() == (1,)
        assert ad._canonical_schema_ready(check, SCHEMA_VERSION)


def test_close_rolls_back_schema_transaction_before_checkpoint(tmp_path, monkeypatch):
    """Close-time TRUNCATE never runs inside an abandoned schema transaction."""
    db_path = tmp_path / "state.db"
    real_connect = sqlite3.connect
    checkpoint_states = []

    class _TransactionTracingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            normalized = sql.lower().replace(" ", "")
            if "wal_checkpoint(" in normalized:
                checkpoint_states.append((normalized, self.in_transaction))
            return super().execute(sql, *args, **kwargs)

    def tracing_connect(*args, **kwargs):
        kwargs["factory"] = _TransactionTracingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", tracing_connect)
    db = SessionDB(db_path=db_path)
    conn = db._conn
    assert conn is not None
    hermes_state._begin_schema_transaction(conn)
    conn.execute("CREATE TABLE partial_schema_marker (id INTEGER)")
    assert conn.in_transaction is True
    assert id(conn) in hermes_state._STATE_SCHEMA_ACTIVE_CONNECTIONS

    checkpoint_states.clear()
    db.close()

    assert checkpoint_states == [("pragmawal_checkpoint(truncate)", False)]
    assert hermes_state._STATE_SCHEMA_ACTIVE_CONNECTIONS == {}
    with real_connect(db_path) as check:
        assert check.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name = 'partial_schema_marker'"
        ).fetchone() == (0,)
