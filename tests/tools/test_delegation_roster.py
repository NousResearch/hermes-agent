"""Privacy and persistence contracts for cross-process delegated-child leases."""

import json
import sqlite3
import threading
import time

import pytest


def test_redact_redacts_before_clipping_a_boundary_secret():
    """A secret split by the display cap cannot expose its leading character."""
    from tools.delegation_roster import _redact

    value = "x" * 487 + " sk-abcdefghijklmnop"

    redacted = _redact(value)

    assert redacted is not None
    assert "sk-abcdefghi" not in redacted


def test_publish_prunes_expired_rows_without_making_list_live_a_writer(tmp_path):
    """Readers only filter expired leases; the next writer reclaims their rows."""
    from tools.delegation_roster import list_live, publish

    home = tmp_path / "profile"
    now = time.time()
    publish({
        "_roster_home": str(home),
        "subagent_id": "sa-live",
        "goal": "live",
        "started_at": now,
    })
    state_db = home / "state.db"
    with sqlite3.connect(state_db) as db:
        db.execute(
            """INSERT INTO delegation_live_subagents
            (subagent_id, depth, goal, started_at, status, tool_count, owner_pid, lease_expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("sa-expired", 0, "expired", now - 10, "running", 0, 0, now - 1),
        )
    before_read = state_db.stat()

    assert [row["subagent_id"] for row in list_live(home)] == ["sa-live"]
    after_read = state_db.stat()
    assert (after_read.st_mtime_ns, after_read.st_size) == (
        before_read.st_mtime_ns,
        before_read.st_size,
    )
    with sqlite3.connect(state_db) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM delegation_live_subagents WHERE subagent_id = ?",
            ("sa-expired",),
        ).fetchone() == (1,)

    publish({
        "_roster_home": str(home),
        "subagent_id": "sa-later",
        "goal": "later",
        "started_at": now,
    })

    with sqlite3.connect(state_db) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM delegation_live_subagents WHERE subagent_id = ?",
            ("sa-expired",),
        ).fetchone() == (0,)


def test_publish_many_persists_a_heartbeat_tick_in_one_transaction(
    tmp_path, monkeypatch
):
    """One heartbeat batch writes every same-profile child through one SQLite transaction."""
    import tools.delegation_roster as roster

    statements = []
    real_open_db = roster.open_db

    def observe_open_db(*args, **kwargs):
        connection = real_open_db(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(roster, "open_db", observe_open_db)
    home = tmp_path / "profile"
    roster.publish_many([
        {"_roster_home": str(home), "subagent_id": "sa-one", "goal": "one"},
        {"_roster_home": str(home), "subagent_id": "sa-two", "goal": "two"},
    ])

    assert [row["subagent_id"] for row in roster.list_live(home)] == [
        "sa-one",
        "sa-two",
    ]
    assert sum(statement.startswith("BEGIN") for statement in statements) == 1
    assert sum(statement.startswith("COMMIT") for statement in statements) == 1


def test_publish_many_skips_a_locked_profile_and_renews_the_next_profile(tmp_path):
    """A locked profile cannot consume another profile's 10-second lease budget."""
    import tools.delegation_roster as roster

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    record_a = {"_roster_home": str(home_a), "subagent_id": "a", "goal": "A"}
    record_b = {"_roster_home": str(home_b), "subagent_id": "b", "goal": "B"}
    roster.publish_many([record_a, record_b])
    with sqlite3.connect(home_b / "state.db") as db:
        db.execute(
            "UPDATE delegation_live_subagents SET lease_expires_at = ? WHERE subagent_id = ?",
            (time.time() + 0.5, "b"),
        )

    blocker = sqlite3.connect(home_a / "state.db", isolation_level=None)
    try:
        blocker.execute("PRAGMA journal_mode=DELETE")
        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            roster.publish_many([record_a, record_b])
        elapsed = time.monotonic() - started

        assert [row["subagent_id"] for row in roster.list_live(home_b)] == ["b"]
        assert elapsed < 2.0
    finally:
        blocker.rollback()
        blocker.close()


def test_publish_many_secures_the_database_after_writer_open_fails(
    tmp_path, monkeypatch
):
    """A failed profile writer still runs the state-file permission repair."""
    import tools.delegation_roster as roster

    path = tmp_path / "profile" / "state.db"
    secured = []

    monkeypatch.setattr(
        roster,
        "_open_writer",
        lambda _path: (_ for _ in ()).throw(sqlite3.OperationalError("locked")),
    )
    monkeypatch.setattr(
        "hermes_state._secure_state_db_files",
        lambda secured_path, **kwargs: secured.append((secured_path, kwargs)),
    )

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        roster.publish_many([
            {
                "_roster_home": str(path.parent),
                "subagent_id": "locked",
                "goal": "locked",
            }
        ])

    assert secured == [(path, {"create_main": True}), (path, {})]


def test_publish_many_continues_after_a_profile_cleanup_error(tmp_path, monkeypatch):
    """A cleanup failure is reported only after later profile batches are attempted."""
    import tools.delegation_roster as roster

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    path_a = home_a / "state.db"

    def fail_a_cleanup(path, **kwargs):
        if path == path_a and not kwargs:
            raise OSError("cleanup failed")

    monkeypatch.setattr("hermes_state._secure_state_db_files", fail_a_cleanup)

    with pytest.raises(OSError, match="cleanup failed"):
        roster.publish_many([
            {"_roster_home": str(home_a), "subagent_id": "a", "goal": "A"},
            {"_roster_home": str(home_b), "subagent_id": "b", "goal": "B"},
        ])

    assert [row["subagent_id"] for row in roster.list_live(home_b)] == ["b"]


def test_remove_fails_fast_while_an_exclusive_profile_lock_is_held(
    tmp_path, monkeypatch
):
    """Lease removal cannot stall the roster lock or skip cleanup after an error."""
    import tools.delegation_roster as roster

    home = tmp_path / "profile"
    record = {"_roster_home": str(home), "subagent_id": "locked", "goal": "locked"}
    roster.publish(record)
    path = home / "state.db"
    secured = []
    monkeypatch.setattr(
        "hermes_state._secure_state_db_files",
        lambda secured_path, **kwargs: secured.append((secured_path, kwargs)),
    )
    blocker = sqlite3.connect(path, isolation_level=None)
    try:
        blocker.execute("PRAGMA journal_mode=DELETE")
        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            roster.remove(record)

        assert time.monotonic() - started < 2.0
        assert secured == [(path, {}), (path, {})]
    finally:
        blocker.rollback()
        blocker.close()


def test_observable_local_records_are_fail_closed_to_the_requested_profile(
    tmp_path, monkeypatch
):
    """A process hosting A and B never exposes B's local child from an A read."""
    from tools import delegate_tool_registry as registry

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    registry._active_subagents.clear()
    monkeypatch.setattr(
        registry,
        "_active_subagents",
        {
            "a": {"subagent_id": "a", "goal": "A", "_roster_home": str(home_a)},
            "b": {"subagent_id": "b", "goal": "B", "_roster_home": str(home_b)},
            # An old/unbound record must not become visible in either profile.
            "unknown": {"subagent_id": "unknown", "goal": "unknown"},
        },
    )
    monkeypatch.setattr("tools.delegation_roster.list_live", lambda home: [])

    assert [
        row["subagent_id"] for row in registry.list_observable_subagents(home_a)
    ] == ["a"]
    assert [
        row["subagent_id"] for row in registry.list_observable_subagents(home_b)
    ] == ["b"]


def test_heartbeat_snapshot_uses_the_record_owner_profile_for_redaction(
    tmp_path, monkeypatch
):
    """A bare heartbeat thread cannot replace B's redaction with launch-profile plaintext."""
    from tools import delegate_tool_registry as registry
    import tools.delegation_roster as roster

    launch, secondary = tmp_path / "launch", tmp_path / "secondary"
    secret = "secondary-only-vault-secret"
    seen_homes = []

    def redact(value, *, force):
        from hermes_constants import get_hermes_home

        seen_homes.append(get_hermes_home().resolve())
        return str(value).replace(secret, "***")

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("agent.redact.redact_sensitive_text", redact)

    record = {"_roster_home": str(secondary), "subagent_id": "b", "goal": secret}
    roster.publish(record)
    with registry._active_subagents_lock:
        registry._active_subagents.clear()
        registry._active_subagents["b"] = record
    heartbeat = threading.Thread(target=registry._heartbeat_publish_once)
    heartbeat.start()
    heartbeat.join(2)

    assert seen_homes and set(seen_homes) == {secondary.resolve()}
    assert secret not in (secondary / "state.db").read_bytes().decode("latin1")
    assert roster.list_live(secondary)[0]["goal"] == "***"


def test_second_heartbeat_does_not_replay_state_schema(tmp_path, monkeypatch):
    """Once this process initialized an unchanged state.db, later leases are DML-only."""
    import tools.delegation_roster as roster

    home = tmp_path / "profile"
    roster.publish({"_roster_home": str(home), "subagent_id": "first", "goal": "first"})
    import tools.async_delegation as async_delegation

    calls = []
    real_initialize = async_delegation._initialize_schema

    def observe_initialize(connection):
        calls.append(connection)
        return real_initialize(connection)

    monkeypatch.setattr(async_delegation, "_initialize_schema", observe_initialize)
    roster.publish({
        "_roster_home": str(home),
        "subagent_id": "second",
        "goal": "second",
    })

    assert calls == []


def test_reader_fails_closed_quickly_while_an_exclusive_delete_journal_lock_is_held(
    tmp_path,
):
    """Observer polling does not hold a gateway response hostage behind a writer lock."""
    from tools.delegation_roster import list_live, publish

    home = tmp_path / "profile"
    publish({"_roster_home": str(home), "subagent_id": "locked", "goal": "locked"})
    state_db = home / "state.db"
    blocker = sqlite3.connect(state_db, isolation_level=None)
    try:
        blocker.execute("PRAGMA journal_mode=DELETE")
        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        assert list_live(home) == []
        assert time.monotonic() - started < 1.5
    finally:
        blocker.rollback()
        blocker.close()


def test_list_live_closes_connection_when_reader_pragma_fails(tmp_path, monkeypatch):
    """A read-only setup failure closes the connection before failing closed."""
    import tools.delegation_roster as roster

    home = tmp_path / "profile"
    home.mkdir()
    (home / "state.db").touch()

    class TrackingConnection:
        close_calls = 0

        def execute(self, statement):
            if statement.startswith("PRAGMA busy_timeout"):
                raise sqlite3.OperationalError("malformed database")

        def close(self):
            self.close_calls += 1

    connection = TrackingConnection()
    monkeypatch.setattr(roster.sqlite3, "connect", lambda *args, **kwargs: connection)

    assert roster.list_live(home) == []
    assert connection.close_calls == 1


def test_unregister_wins_over_an_inflight_heartbeat_publish(tmp_path, monkeypatch):
    """A captured heartbeat cannot resurrect a lease after its owner unregisters it."""
    from tools import delegate_tool_registry as registry
    from tools.delegation_roster import list_live

    home = tmp_path / "profile"
    record = {"_roster_home": str(home), "subagent_id": "racing", "goal": "racing"}
    entered, release, unregistered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    real_publish = registry._sync_shared_roster_many

    def blocked_publish(records):
        entered.set()
        assert release.wait(5), "test did not release the heartbeat"
        real_publish(records)

    monkeypatch.setattr(registry, "_sync_shared_roster_many", blocked_publish)
    monkeypatch.setattr(registry, "_ROSTER_HEARTBEAT_SECONDS", 0)
    with registry._active_subagents_lock:
        registry._active_subagents.clear()
        registry._active_subagents["racing"] = record

    heartbeat = threading.Thread(target=registry._run_roster_heartbeat)
    heartbeat.start()
    assert entered.wait(2)
    unregister = threading.Thread(
        target=lambda: (registry._unregister_subagent("racing"), unregistered.set())
    )
    unregister.start()
    assert not unregistered.wait(0.1)
    release.set()
    heartbeat.join(5)
    unregister.join(5)

    assert unregistered.is_set()
    assert list_live(home) == []


def test_lease_tolerates_a_missed_heartbeat_but_expires_a_dead_owner(
    tmp_path, monkeypatch
):
    """A 2.25s scheduler pause stays visible with 1s beats; an unrefreshed lease still expires."""
    import tools.delegation_roster as roster

    home = tmp_path / "profile"
    roster.publish({
        "_roster_home": str(home),
        "subagent_id": "paused",
        "goal": "paused",
    })
    time.sleep(2.25)
    assert [row["subagent_id"] for row in roster.list_live(home)] == ["paused"]

    monkeypatch.setattr(roster, "_LEASE_SECONDS", 0.05)
    roster.publish({"_roster_home": str(home), "subagent_id": "dead", "goal": "dead"})
    time.sleep(0.1)
    assert "dead" not in [row["subagent_id"] for row in roster.list_live(home)]


def test_replaced_database_is_initialized_again_before_writing(tmp_path):
    """The identity cache avoids replay but does not mistake a replacement DB for the old schema."""
    import tools.delegation_roster as roster

    home = tmp_path / "profile"
    roster.publish({"_roster_home": str(home), "subagent_id": "first", "goal": "first"})
    state_db = home / "state.db"
    state_db.unlink()
    roster.publish({
        "_roster_home": str(home),
        "subagent_id": "replacement",
        "goal": "replacement",
    })

    assert [row["subagent_id"] for row in roster.list_live(home)] == ["replacement"]


def test_persisted_observer_rows_never_grant_stop_or_steer_authority(tmp_path):
    """A foreign lease is an observer-only projection, not an addressable control capability."""
    from tools import delegate_tool_registry as registry
    from tools.delegation_roster import list_live, publish

    home = tmp_path / "profile"
    publish({"_roster_home": str(home), "subagent_id": "foreign", "goal": "view only"})
    with registry._active_subagents_lock:
        registry._active_subagents.clear()

    assert list_live(home)[0]["subagent_id"] == "foreign"
    assert registry.interrupt_subagent("foreign") is False
    assert registry.steer_subagent("foreign", "stop") is False


def test_symlinked_vault_scope_redacts_thread_heartbeat_storage_and_rpc(
    tmp_path, monkeypatch
):
    """Alias-registered exact values cannot escape the canonical roster owner scope."""
    from agent import redact
    from tools import delegate_tool_registry as registry
    from tui_gateway import server

    real_home = tmp_path / "real-home"
    alias_home = tmp_path / "home-alias"
    real_home.mkdir()
    try:
        alias_home.symlink_to(real_home, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    canary = "vault orchard quokka hummingbird 482"
    assert redact.redact_sensitive_text(canary, force=True) == canary

    monkeypatch.setattr(registry, "_active_subagents", {})
    monkeypatch.setattr(registry, "_ensure_roster_heartbeat", lambda: None)
    try:
        monkeypatch.setenv("HERMES_HOME", str(alias_home))
        redact.register_vault_redaction_value(canary)
        registry._register_subagent({"subagent_id": "alias-canary", "goal": canary})

        heartbeat = threading.Thread(target=registry._heartbeat_publish_once)
        heartbeat.start()
        heartbeat.join(2)
        assert not heartbeat.is_alive()

        state_db = real_home / "state.db"
        assert canary.encode() not in state_db.read_bytes()
        with sqlite3.connect(state_db) as db:
            stored = db.execute(
                "SELECT goal FROM delegation_live_subagents WHERE subagent_id = ?",
                ("alias-canary",),
            ).fetchone()
        assert stored is not None and canary not in stored[0]

        monkeypatch.setenv("HERMES_HOME", str(real_home))
        reply = server.dispatch({"id": 1, "method": "delegation.status", "params": {}})
        rendered = json.dumps(reply["result"]["active"], ensure_ascii=False)
        assert canary not in rendered
        assert "«redacted-vault-secret»" in rendered
    finally:
        registry._unregister_subagent("alias-canary")
        for home in (alias_home, real_home):
            monkeypatch.setenv("HERMES_HOME", str(home))
            redact.clear_vault_redaction_values()
