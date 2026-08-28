"""Stage-1 tests for the unreachable task-scoped closeout ledger."""

import importlib
import json
import queue
import sqlite3
import threading
import time

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools import async_delegation as ad


ENABLED = {"delegation": {"task_scoped_closeout": True}}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in tuple(__import__("os").environ):
        if key.startswith("API_SERVER_"):
            monkeypatch.delenv(key, raising=False)
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _register(work="work-1", member="deleg-1", turn="turn-1", **kwargs):
    aggregate_char_budget = kwargs.pop("aggregate_char_budget", None)
    return ad.register_work_group_member(
        work_id=work,
        owner_turn_id=turn,
        delegation_id=member,
        feature_config=ENABLED,
        routing={"origin_session": "route", "parent_session_id": "parent"},
        task={"goal": member, "task_index": kwargs.pop("task_index", 0), **kwargs},
        aggregate_char_budget=aggregate_char_budget,
    )


def _finish(member, *, status="completed", summary="done", **metadata):
    result = {"status": status, "summary": summary, **metadata}
    event = {"delegation_id": member, "status": status, "completed_at": time.time()}
    return ad.persist_group_member_completion(member, event, result)


def _claim_bound(work="work-1", turn="closeout-1"):
    claimed = ad.claim_ready_work_group(work, "test")
    assert claimed is not None
    envelope = claimed["envelope"]
    assert ad.bind_work_group_closeout_turn(
        work, envelope["delivery_id"], claimed["claim_id"], turn
    )
    return claimed


def test_creation_gate_defaults_false_and_default_is_canonical():
    assert DEFAULT_CONFIG["delegation"]["task_scoped_closeout"] is False
    assert ad.task_scoped_closeout_enabled({}) is False
    assert not ad.register_work_group_member(
        work_id="off", owner_turn_id="t", delegation_id="d", feature_config={}
    )
    with ad._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM async_delegation_work_groups"
        ).fetchone()[0] == 0


def test_session_boundary_close_is_scoped_idempotent_and_clears_ownership():
    for work, route, parent, member in (
        ("work-a", "route-a", "session-a", "delegate-a"),
        ("work-b", "route-b", "session-b", "delegate-b"),
    ):
        assert ad.register_work_group_member(
            work_id=work,
            owner_turn_id="owner",
            delegation_id=member,
            feature_config=ENABLED,
            routing={"origin_session": route, "parent_session_id": parent},
            task={"goal": member, "task_index": 0},
        )
    _finish("delegate-a")
    assert ad.seal_work_group("work-a", "owner")
    _claim_bound("work-a", "closeout-a")

    assert ad.close_work_groups_for_session(
        parent_session_id="session-a",
        diagnostics="reset session-a",
    ) == 1
    assert ad.close_work_groups_for_session(
        parent_session_id="session-a",
        diagnostics="reset session-a",
    ) == 0

    with ad._transaction() as conn:
        conn.row_factory = sqlite3.Row
        rows = {
            row["work_id"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM async_delegation_work_groups ORDER BY work_id"
            ).fetchall()
        }
    closed = rows["work-a"]
    assert closed["state"] == "closed"
    assert closed["terminal_disposition"] == "cancelled"
    assert closed["terminal_diagnostics"] == "reset session-a"
    for column in (
        "closeout_claim", "closeout_claimed_at", "closeout_turn_id",
        "closeout_owner_pid", "closeout_owner_started_at",
    ):
        assert closed[column] is None
    assert rows["work-b"]["state"] == "open"


def test_session_boundary_preserves_legacy_empty_work_id_row():
    now = time.time()
    with ad._transaction() as conn:
        conn.execute(
            "INSERT INTO async_delegation_work_groups "
            "(work_id,origin_session,owner_turn_id,state,created_at,updated_at) "
            "VALUES ('','legacy-route','owner','open',?,?)",
            (now, now),
        )

    assert ad.close_work_groups_for_session(origin_session="legacy-route") == 0
    with ad._transaction() as conn:
        row = conn.execute(
            "SELECT state,terminal_disposition FROM async_delegation_work_groups "
            "WHERE work_id=''"
        ).fetchone()
    assert tuple(row) == ("open", None)


def test_closed_group_recovery_reveals_crash_window_provisional():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    claimed = _claim_bound(turn="closeout-1")
    envelope = claimed["envelope"]
    assert ad.close_work_group(
        "work-1",
        0,
        envelope["delivery_id"],
        claimed["claim_id"],
        "closeout-1",
    )
    metadata = json.dumps({
        "hidden": True,
        "work_id": "work-1",
        "delivery_id": envelope["delivery_id"],
    })
    with ad._transaction() as conn:
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, "
            "display_kind TEXT, display_metadata TEXT)"
        )
        conn.execute(
            "INSERT INTO messages VALUES (1, ?, ?)",
            ("delegation_closeout_provisional", metadata),
        )
        conn.execute(
            "INSERT INTO messages VALUES (2, ?, ?)",
            ("delegation_closeout_provisional", metadata),
        )

    assert ad.reconcile_closed_closeout_provisionals() == 1
    with ad._transaction() as conn:
        rows = conn.execute(
            "SELECT display_kind, display_metadata FROM messages ORDER BY id"
        ).fetchall()
    assert rows == [
        (None, None),
        ("delegation_closeout_provisional", metadata),
    ]


def test_old_async_delegation_schema_is_upgraded_additively_and_legacy_restores(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    now = time.time()
    conn.execute(
        """CREATE TABLE async_delegations (
        delegation_id TEXT PRIMARY KEY, origin_session TEXT NOT NULL,
        origin_ui_session_id TEXT NOT NULL DEFAULT '', parent_session_id TEXT,
        state TEXT NOT NULL, dispatched_at REAL NOT NULL, completed_at REAL,
        updated_at REAL NOT NULL, event_json TEXT, result_json TEXT,
        delivery_state TEXT NOT NULL DEFAULT 'pending',
        delivery_attempts INTEGER NOT NULL DEFAULT 0, delivered_at REAL)"""
    )
    event = {"type": "async_delegation", "delegation_id": "legacy"}
    conn.execute(
        """INSERT INTO async_delegations
        (delegation_id,origin_session,state,dispatched_at,completed_at,updated_at,
         event_json,delivery_state) VALUES ('legacy','','completed',1,2,2,?,'pending')""",
        (json.dumps(event),),
    )
    conn.execute(
        "UPDATE async_delegations SET dispatched_at=?, completed_at=?, updated_at=?",
        (now - 2, now - 1, now - 1),
    )
    conn.commit()
    conn.close()

    target = queue.Queue()
    assert ad.restore_undelivered_completions(target) == 1
    assert target.get_nowait()["delegation_id"] == "legacy"
    with ad._transaction() as upgraded:
        cols = {row[1] for row in upgraded.execute("PRAGMA table_info(async_delegations)")}
        assert {"origin_work_id", "work_generation"} <= cols


def test_open_membership_blocks_claim_until_owner_seals():
    assert _register(member="a", task_index=0)
    assert _finish("a") is False
    assert _register(member="b", task_index=1)
    assert ad.claim_ready_work_group("work-1", "early") is None
    assert ad.seal_work_group("work-1", "wrong-turn") is False
    assert ad.seal_work_group("work-1", "turn-1") is True
    assert ad.claim_ready_work_group("work-1", "still-pending") is None
    assert _finish("b") is True


def test_seal_and_last_completion_race_is_ready_and_duplicate_safe():
    assert _register()
    barrier = threading.Barrier(2)
    outcomes = []

    def seal():
        barrier.wait()
        outcomes.append(ad.seal_work_group("work-1", "turn-1"))

    def finish():
        barrier.wait()
        outcomes.append(_finish("deleg-1"))

    threads = [threading.Thread(target=seal), threading.Thread(target=finish)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert any(outcomes)
    assert ad.group_is_ready("work-1")
    first = ad.claim_ready_work_group("work-1", "one")
    assert first is not None
    assert ad.claim_ready_work_group("work-1", "two") is None


def test_exactly_one_concurrent_claim_winner():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    barrier = threading.Barrier(8)
    claims = []

    def racer(index):
        barrier.wait()
        claims.append(ad.claim_ready_work_group("work-1", f"c{index}"))

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(claim is not None for claim in claims) == 1


def test_dead_unbound_queue_owner_is_reclaimed_without_stale_delay():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = ad.claim_ready_work_group("work-1", "first")
    assert old is not None
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_work_groups "
            "SET closeout_owner_pid=2147483647, closeout_owner_started_at=1"
        )

    new = ad.reclaim_stale_work_group_claim("work-1", "restart")

    assert new is not None
    assert new["claim_id"] != old["claim_id"]
    assert new["envelope"] == old["envelope"]


def test_delivery_and_envelope_survive_release_and_module_style_reload():
    assert _register()
    _finish("deleg-1", summary="stable")
    ad.seal_work_group("work-1", "turn-1")
    claimed = ad.claim_ready_work_group("work-1", "first")
    assert claimed is not None
    assert ad.release_work_group_claim("work-1", claimed["claim_id"])
    expected = claimed["envelope"]

    reloaded = importlib.reload(ad)
    recovered = reloaded.recover_work_groups()
    assert recovered[0]["delivery_id"] == expected["delivery_id"]
    assert recovered[0]["envelope"] == expected
    reclaimed = reloaded.reclaim_stale_work_group_claim("work-1", "restart")
    assert reclaimed is not None
    assert reclaimed["envelope"] == expected


def test_live_bound_claim_is_not_stolen_after_age_and_can_heartbeat():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    claimed = _claim_bound()
    delivery = claimed["envelope"]["delivery_id"]
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_work_groups SET closeout_claimed_at=0"
        )
    assert ad.reclaim_stale_work_group_claim("work-1", "thief") is None
    assert not ad.release_work_group_claim("work-1", claimed["claim_id"])
    assert ad.renew_work_group_claim(
        "work-1", 0, delivery, claimed["claim_id"], "closeout-1"
    )


def test_failed_live_bound_turn_releases_exact_identity_for_recovery():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = _claim_bound()
    delivery = old["envelope"]["delivery_id"]

    assert ad.release_bound_work_group_closeout(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.close_work_group(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id=delivery,
        claim_id=old["claim_id"], closeout_turn_id="closeout-1",
        delegation_id="stale-replacement",
    )
    recovered = ad.reclaim_stale_work_group_claim("work-1", "recovery")
    assert recovered is not None
    assert recovered["envelope"] == old["envelope"]
    assert recovered["claim_id"] != old["claim_id"]


def test_exact_release_cannot_clear_successor_binding():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = _claim_bound()
    delivery = old["envelope"]["delivery_id"]
    assert ad.release_bound_work_group_closeout(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    successor = ad.reclaim_stale_work_group_claim("work-1", "successor")
    assert successor is not None
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, successor["claim_id"], "closeout-2"
    )

    assert not ad.release_bound_work_group_closeout(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert ad.renew_work_group_claim(
        "work-1", 0, delivery, successor["claim_id"], "closeout-2"
    )


def test_dead_bound_claim_rotates_and_stale_actor_cannot_close_or_reopen():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = _claim_bound()
    delivery = old["envelope"]["delivery_id"]
    with ad._transaction() as conn:
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claimed_at=0, closeout_owner_pid=2147483647,
                   closeout_owner_started_at=1"""
        )
    new = ad.reclaim_stale_work_group_claim("work-1", "recovery")
    assert new is not None and new["claim_id"] != old["claim_id"]
    assert new["envelope"]["delivery_id"] == delivery
    assert not ad.close_work_group(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id=delivery,
        claim_id=old["claim_id"], closeout_turn_id="closeout-1",
        delegation_id="stale-replacement",
    )
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, new["claim_id"], "closeout-2"
    )


def test_enqueue_deduplication_is_scoped_by_profile_home(tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    target = queue.Queue()
    base_claim = {
        "envelope": {
            "type": "async_delegation_work_closeout",
            "work_id": "same-work",
            "generation": 0,
            "delivery_id": "same-delivery",
        },
        "routing": {},
    }
    for profile, claim_id in (("profile-a", "claim-a"), ("profile-b", "claim-b")):
        home = tmp_path / profile
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            claimed = {**base_claim, "claim_id": claim_id}
            assert ad._enqueue_claimed_work_group(claimed, target_queue=target)
        finally:
            reset_hermes_home_override(token)

    first = target.get_nowait()
    second = target.get_nowait()
    assert first["delivery_id"] == second["delivery_id"] == "same-delivery"
    assert first["_ledger_profile_home"] != second["_ledger_profile_home"]
    assert target.empty()


def test_process_registry_startup_recovers_group_with_creation_disabled():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = ad.claim_ready_work_group("work-1", "pre-crash")
    assert old is not None
    discarded = queue.Queue()
    assert ad._enqueue_claimed_work_group(old, target_queue=discarded) is not None
    discarded.get_nowait()
    with ad._transaction() as conn:
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claimed_at=0, closeout_owner_pid=2147483647,
                   closeout_owner_started_at=1"""
        )

    # A new host starts with the creation flag still absent/false. Registry
    # construction must nevertheless recover the existing durable group onto
    # that host's own completion rail without importing the module singleton.
    ad._reset_for_tests()
    import tools.process_registry as pr_module

    # Importing the module creates its singleton; if the module was already
    # imported earlier in this pytest process, constructing a fresh registry is
    # the equivalent startup boundary for this isolated ledger.
    registry = pr_module.process_registry
    if registry.completion_queue.empty():
        registry = pr_module.ProcessRegistry()
    recovered = registry.completion_queue.get_nowait()
    assert recovered["type"] == "async_delegation_work_closeout"
    assert recovered["origin_work_id"] == "work-1"
    assert recovered["claim_id"] != old["claim_id"]
    assert ad.task_scoped_closeout_enabled({}) is False


def test_recovery_rotates_dead_bound_claim_before_enqueueing_replacement():
    assert _register()
    _finish("deleg-1")
    assert ad.seal_work_group("work-1", "turn-1")
    old = _claim_bound()
    delivery = old["envelope"]["delivery_id"]
    with ad._transaction() as conn:
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claimed_at=0, closeout_owner_pid=2147483647,
                   closeout_owner_started_at=1"""
        )

    events = ad.recover_and_enqueue_work_groups(consumer="restart")

    assert len(events) == 1
    recovered = events[0]
    assert recovered["delivery_id"] == delivery
    assert recovered["claim_id"] != old["claim_id"]
    assert ad.bind_work_group_closeout_turn(
        "work-1", delivery, recovered["claim_id"], "closeout-2"
    )
    assert not ad.renew_work_group_claim(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.release_bound_work_group_closeout(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.close_work_group(
        "work-1", 0, delivery, old["claim_id"], "closeout-1"
    )
    assert not ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id=delivery,
        claim_id=old["claim_id"], closeout_turn_id="closeout-1",
        delegation_id="stale-replacement",
    )
    assert ad.close_work_group(
        "work-1", 0, delivery, recovered["claim_id"], "closeout-2"
    )
    assert ad.recover_and_enqueue_work_groups(consumer="restart-again") == []


def test_bounded_aggregate_has_deterministic_order_and_terminal_metadata():
    assert _register(member="later", task_index=2)
    assert _register(member="first", task_index=0)
    _finish("later", summary="L" * 10_000, status="timed_out", timeout_seconds=30)
    _finish(
        "first", summary="F" * 10_000, status="stalled",
        stalled_after_quiet_seconds=450, schema_verdict={"valid": False},
    )
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_work_groups SET aggregate_char_budget=1400 "
            "WHERE work_id='work-1'"
        )
    ad.seal_work_group("work-1", "turn-1")
    claimed = ad.claim_ready_work_group("work-1", "bounded")
    envelope = claimed["envelope"]
    assert [m["delegation_id"] for m in envelope["members"]] == ["first", "later"]
    assert envelope["members"][0]["stalled_after_quiet_seconds"] == 450
    assert envelope["members"][0]["schema_verdict"] == {"valid": False}
    assert envelope["members"][1]["timeout_seconds"] == 30
    assert all(m["detail_ref"].startswith("async_delegation:") for m in envelope["members"])
    assert len(ad._json_bytes(envelope)) <= 1400


def test_total_byte_budget_handles_multibyte_and_untrusted_nested_metadata():
    assert _register(member="多字节", goal="界" * 20_000)
    _finish(
        "多字节", summary="🙂" * 20_000, error="錯" * 20_000,
        schema_valid=False, schema_errors=["壊" * 10_000] * 20,
        schema_retries=3, schema_verdict={"valid": False, "reason": "悪" * 20_000},
    )
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_work_groups SET aggregate_char_budget=1024"
        )
    assert ad.seal_work_group("work-1", "turn-1")
    envelope = ad.claim_ready_work_group("work-1", "bounded")["envelope"]
    assert len(ad._json_bytes(envelope)) <= 1024
    member = envelope["members"][0]
    assert member["delegation_id"] == "多字节"
    assert member["status"] == "completed"
    assert member["detail_ref"] == "async_delegation:多字节"


def test_registration_stops_before_minimal_member_set_cannot_fit():
    admitted = []
    for index in range(100):
        member = f"member-{index:03d}-" + "x" * 20
        if not _register(member=member, task_index=index, aggregate_char_budget=1024):
            break
        admitted.append(member)
    assert admitted and len(admitted) < 100
    for member in admitted:
        _finish(member)
    assert ad.seal_work_group("work-1", "turn-1")
    first = ad.claim_ready_work_group("work-1", "one")["envelope"]
    assert {member["delegation_id"] for member in first["members"]} == set(admitted)
    assert len(ad._json_bytes(first)) <= 1024
    with ad._transaction() as conn:
        persisted = conn.execute(
            "SELECT closeout_payload_json FROM async_delegation_work_groups"
        ).fetchone()[0]
    assert persisted.encode("utf-8") == ad._json_bytes(first)


def test_admission_boundary_survives_multibyte_untrusted_outcomes_and_reload():
    statuses = ("completed", "FAILED", "timed-out", "stalled", "取消🙂" * 40)
    admitted = []
    for index in range(100):
        member = f"成员-{index:03d}-" + "界" * 8
        if not _register(member=member, task_index=index, aggregate_char_budget=3000):
            break
        admitted.append(member)
    assert admitted and len(admitted) < 100

    for index, member in enumerate(admitted):
        status = statuses[index % len(statuses)]
        _finish(
            member,
            status=status,
            summary="概" * 10_000,
            error="錯" * 10_000,
            diagnostic="診" * 10_000,
            schema_errors=["壊" * 10_000] * 10,
            schema_retries=10**100,
            schema_verdict={"valid": False, "reason": "悪" * 10_000},
        )
    with ad._transaction() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM async_delegations WHERE origin_work_id='work-1'"
        ).fetchall()
    delivery_id = ad._delivery_id("work-1", 0)
    forward = ad._build_group_envelope("work-1", 0, delivery_id, 3000, rows)
    reverse = ad._build_group_envelope("work-1", 0, delivery_id, 3000, list(reversed(rows)))
    assert ad._json_bytes(forward) == ad._json_bytes(reverse)
    assert ad.seal_work_group("work-1", "turn-1")
    claimed = ad.claim_ready_work_group("work-1", "boundary")
    assert claimed is not None
    envelope = claimed["envelope"]
    ids = [member["delegation_id"] for member in envelope["members"]]
    assert sorted(ids) == sorted(admitted)
    assert len(ids) == len(set(ids))
    assert len(ad._json_bytes(envelope)) <= 3000
    expected = {
        member: ad._status_category(statuses[index % len(statuses)])
        for index, member in enumerate(admitted)
    }
    assert {item["delegation_id"]: item["status"] for item in envelope["members"]} == expected
    assert all(item["detail_ref"] == f"async_delegation:{item['delegation_id']}"
               for item in envelope["members"])
    assert all(item["detail_truncated"] for item in envelope["members"])

    expected_bytes = ad._json_bytes(envelope)
    reloaded = importlib.reload(ad)
    recovered = reloaded.recover_work_groups()
    assert reloaded._json_bytes(recovered[0]["envelope"]) == expected_bytes


@pytest.mark.parametrize(
    ("field", "kwargs"),
    (
        ("work_id", {"work": "界" * 86}),
        ("delegation_id", {"member": "界" * 86}),
        ("owner_turn_id", {"turn": "界" * 86}),
    ),
)
def test_overlong_utf8_identifier_is_rejected_without_leaving_group(field, kwargs):
    assert not _register(**kwargs), field
    with ad._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM async_delegation_work_groups"
        ).fetchone()[0] == 0


def test_reopen_requires_trusted_identity_and_old_generation_cannot_mutate_new():
    assert _register()
    _finish("deleg-1")
    ad.seal_work_group("work-1", "turn-1")
    claimed = _claim_bound()
    delivery = claimed["envelope"]["delivery_id"]
    assert not ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id="wrong",
        claim_id=claimed["claim_id"],
        closeout_turn_id="closeout-1", delegation_id="replacement"
    )
    assert ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id=delivery,
        claim_id=claimed["claim_id"],
        closeout_turn_id="closeout-1", delegation_id="replacement"
    )
    assert not ad.close_work_group(
        "work-1", 0, delivery, claimed["claim_id"], "closeout-1"
    )
    assert not ad.reopen_work_group_with_member(
        work_id="work-1", generation=0, delivery_id=delivery,
        claim_id=claimed["claim_id"],
        closeout_turn_id="closeout-1", delegation_id="another"
    )
    _finish("replacement")
    assert ad.seal_work_group("work-1", "closeout-1")
    next_claim = _claim_bound(turn="closeout-2")
    next_delivery = next_claim["envelope"]["delivery_id"]
    assert next_delivery != delivery
    assert not ad.close_work_group(
        "work-1", 1, delivery, next_claim["claim_id"], "closeout-2"
    )
    assert ad.close_work_group(
        "work-1", 1, next_delivery, next_claim["claim_id"], "closeout-2"
    )


def test_unresolved_grouped_rows_survive_legacy_pruning_age_and_attempt_drop(monkeypatch):
    assert _register()
    _finish("deleg-1")
    with ad._transaction() as conn:
        conn.execute(
            """UPDATE async_delegations SET updated_at=0, completed_at=0,
               delivery_attempts=999, delivery_claim='legacy-claim'
               WHERE delegation_id='deleg-1'"""
        )
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 0)
    monkeypatch.setattr(ad, "_MAX_DURABLE_PENDING", 0)
    ad._prune_durable_records()
    assert not ad.release_completion_delivery("deleg-1", "legacy-claim")
    assert not ad.complete_completion_delivery("deleg-1", "legacy-claim")
    assert not ad.drop_completion_delivery("deleg-1", "anything")
    target = queue.Queue()
    assert ad.restore_undelivered_completions(target) == 0
    with ad._transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id='deleg-1'"
        ).fetchone()
    assert row == ("pending",)


def test_old_or_oversized_terminal_detail_becomes_honest_tombstone(monkeypatch):
    assert _register()
    _finish("deleg-1", summary="z" * 100_000, error="e" * 100_000)
    with ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET updated_at=0")
    monkeypatch.setattr(ad, "_MAX_GROUP_DETAIL_BYTES", 100)
    ad._prune_durable_records()
    with ad._transaction() as conn:
        member = conn.execute(
            "SELECT state,result_json FROM async_delegations"
        ).fetchone()
        group = conn.execute(
            "SELECT state,terminal_diagnostics FROM async_delegation_work_groups"
        ).fetchone()
    tombstone = json.loads(member[1])
    assert member[0] == "completed" and tombstone["detail_lost"] is True
    assert tombstone["status"] == "completed"
    assert group[0] == "open" and "unknown" in group[1]
    assert ad.seal_work_group("work-1", "turn-1")
    envelope = ad.claim_ready_work_group("work-1", "tombstone")["envelope"]
    assert envelope["members"][0]["detail_lost"] is True
    assert "compacted" in envelope["members"][0]["diagnostic"]


def test_first_member_collision_does_not_leave_empty_group():
    with ad._transaction() as conn:
        now = time.time()
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id,origin_session,state,dispatched_at,updated_at)
               VALUES ('collision','','completed',?,?)""",
            (now, now),
        )
    assert not _register(work="new-work", member="collision")
    with ad._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM async_delegation_work_groups WHERE work_id='new-work'"
        ).fetchone()[0] == 0


def test_dead_open_owner_becomes_explicit_unknown_and_ready():
    assert _register()
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_work_groups SET owner_pid=2147483647, owner_started_at=1"
        )
        conn.execute(
            "UPDATE async_delegations SET owner_pid=2147483647, owner_started_at=1"
        )
    recovered = ad.recover_work_groups()
    assert recovered == [{"work_id": "work-1", "state": "sealed_ready"}]
    with ad._transaction() as conn:
        group = conn.execute(
            "SELECT state,terminal_diagnostics FROM async_delegation_work_groups"
        ).fetchone()
        member = conn.execute(
            "SELECT state,result_json FROM async_delegations"
        ).fetchone()
    assert group[0] == "sealed" and "unknown" in group[1]
    assert member[0] == "unknown"
    assert json.loads(member[1])["status"] == "unknown"
