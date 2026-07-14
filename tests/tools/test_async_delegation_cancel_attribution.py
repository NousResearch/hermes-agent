"""Phase-0 cancel forensics: every ``state=cancelled`` writer stamps WHO/WHY/WHEN.

Parent-interrupt spec Phase-0 (2026-07-13): observation-only instrumentation so
a silently-dead background delegation record names its killer. No cancellation
behavior changes — these tests assert attribution is stamped on every terminal
path, absent on healthy records, and that the schema stays additive
(pre-upgrade records still parse and old readers ignore the new keys).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from tools import async_delegation as ad
from tools import async_delegation_store as store
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield tmp_path
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _spec():
    return {
        "profile": "default",
        "source": {
            "kind": "single",
            "tasks": [{
                "goal": "continue the report",
                "context": "write /tmp/report.md",
                "role": "leaf",
                "inherit_context": False,
            }],
            "shared_context": None,
        },
        "execution": {
            "model": "test-model",
            "provider": "test-provider",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
            "acp_command": None,
            "acp_args": [],
            "reasoning_config": None,
            "fallback_chain": None,
            "service_tier": None,
            "provider_preferences": None,
            "toolsets": ["file"],
            "max_iterations": 50,
            "parent_depth": 0,
            "workspace_hint": "/tmp",
            "credential_ref": {"provider": "test-provider", "custom_provider": None},
        },
        "route": {
            "session_key": "agent:main:telegram:dm:123",
            "parent_session_id": "parent-1",
            "origin_ui_session_id": "",
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "thread_id": None,
            "user_id": "u1",
            "user_name": "Ace",
            "profile": "default",
        },
    }


def _dispatch(gate=None):
    gate = gate or threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    result = ad.dispatch_async_delegation_batch(
        goals=["continue the report"],
        context="write /tmp/report.md",
        toolsets=["file"],
        role="leaf",
        model="test-model",
        session_key="agent:main:telegram:dm:123",
        parent_session_id="parent-1",
        runner=runner,
        max_async_children=3,
        durable_spec=_spec(),
        current_boot_id="100:1.0",
    )
    return result, gate


def _load():
    return json.loads(ad._registry_path().read_text(encoding="utf-8"))


def _legacy_record(delegation_id="deleg_legacy", state="running"):
    """A pre-upgrade v1 record: no lifecycle trail, no cancel_attribution."""
    now = time.time()
    spec = _spec()
    return {
        "delegation_id": delegation_id,
        "state": state,
        "created_at": now,
        "updated_at": now,
        "profile": "default",
        "source": spec["source"],
        "execution": spec["execution"],
        "route": spec["route"],
        "attempt": {
            "attempt_id": f"{delegation_id}:g0:old",
            "generation": 0,
            "redispatch_count": 0,
            "owner_boot_id": "100:1.0",
            "started_at": now,
            "submitted_at": 1.0,
            "last_interrupted_at": None,
            "last_error": None,
        },
        "terminal": None,
        "outbox": [],
    }


def _write_record(record):
    state = {
        "schema_version": 1,
        "updated_at": time.time(),
        "records": {record["delegation_id"]: record},
    }
    ad._write_registry_for_tests(state)


# --- terminal paths stamp attribution -------------------------------------

def test_session_end_cancel_stamps_who_why_when():
    result, gate = _dispatch()
    before = time.time()
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "cancelled"
    attribution = record["cancel_attribution"]
    assert attribution["reason"] == "session_end"                 # WHY
    assert attribution["via"] == "cancel_matching"
    assert before <= attribution["cancelled_at"] <= time.time()   # WHEN
    # WHO: the immediate caller frame plus a captured stack ending at the
    # runtime code that invoked interrupt_for_session.
    assert ":" in attribution["caller"]
    assert attribution["stack"], "caller stack must be captured"
    assert any("interrupt_for_session" in frame for frame in attribution["stack"])
    assert attribution["selector"] == {"parent_session_id": "parent-1"}
    gate.set()


def test_stop_command_reason_flows_through_interrupt_for_session():
    result, gate = _dispatch()
    ad.interrupt_for_session(
        session_key="agent:main:telegram:dm:123",
        parent_session_id="parent-1",
        reason="stop_command",
    )
    attribution = _load()["records"][result["delegation_id"]]["cancel_attribution"]
    assert attribution["reason"] == "stop_command"
    assert attribution["selector"]["session_key"] == "agent:main:telegram:dm:123"
    gate.set()


def test_session_reset_reason_is_recorded_distinctly():
    result, gate = _dispatch()
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_reset")
    attribution = _load()["records"][result["delegation_id"]]["cancel_attribution"]
    assert attribution["reason"] == "session_reset"
    gate.set()


def test_interrupt_all_stop_stamps_attribution_with_all_active_selector():
    result, gate = _dispatch()
    ad.interrupt_all(reason="/stop")
    record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "cancelled"
    attribution = record["cancel_attribution"]
    assert attribution["reason"] == "/stop"
    assert attribution["via"] == "cancel_matching"
    assert attribution["selector"] == {"all_active": True}
    assert any("interrupt_all" in frame for frame in attribution["stack"])
    gate.set()


def test_shutdown_recoverable_leaves_no_cancel_attribution_but_breadcrumbs():
    result, gate = _dispatch()
    ad.interrupt_all(reason="gateway shutdown (phase)", recoverable=True)
    record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "recoverable"
    assert "cancel_attribution" not in record
    events = [e["event"] for e in record["lifecycle"]]
    assert "recoverable" in events
    recoverable_event = next(e for e in record["lifecycle"] if e["event"] == "recoverable")
    assert "gateway shutdown (phase)" in recoverable_event["detail"]
    gate.set()


def test_resume_disabled_durable_only_cancel_is_attributed():
    record = _legacy_record()
    record["state"] = "recoverable"
    _write_record(record)
    assert ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end") == 0
    stored = _load()["records"][record["delegation_id"]]
    assert stored["state"] == "cancelled"
    assert stored["cancel_attribution"]["reason"] == "session_end"


# --- healthy records carry no attribution ----------------------------------

def test_uncancelled_running_record_has_no_attribution():
    result, gate = _dispatch()
    record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "running"
    assert "cancel_attribution" not in record
    gate.set()


def test_completed_record_has_no_attribution_and_terminal_breadcrumb():
    result, gate = _dispatch()
    gate.set()
    record = _load()["records"][result["delegation_id"]]
    deadline = time.time() + 5
    while time.time() < deadline and record["state"] == "running":
        time.sleep(0.05)
        record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "done"
    assert "cancel_attribution" not in record
    events = [e["event"] for e in record["lifecycle"]]
    assert events[0] == "spawned"
    assert "done" in events


# --- lifecycle breadcrumb trail --------------------------------------------

def test_lifecycle_trail_orders_spawned_running_cancelled():
    result, gate = _dispatch()
    # mark_submitted (the "running" breadcrumb) runs on the worker thread;
    # wait for it so the cancel deterministically lands third in the trail.
    deadline = time.time() + 5
    record = _load()["records"][result["delegation_id"]]
    while time.time() < deadline and not record["attempt"].get("submitted_at"):
        time.sleep(0.02)
        record = _load()["records"][result["delegation_id"]]
    assert record["attempt"].get("submitted_at"), "worker never marked submitted"
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    record = _load()["records"][result["delegation_id"]]
    events = [e["event"] for e in record["lifecycle"]]
    assert events[0] == "spawned"
    assert "running" in events
    assert events[-1] == "cancelled"
    cancelled = record["lifecycle"][-1]
    assert "reason=session_end" in cancelled["detail"]
    assert all(
        earlier["ts"] <= later["ts"]
        for earlier, later in zip(record["lifecycle"], record["lifecycle"][1:])
    )
    gate.set()


def test_lifecycle_trail_is_capped():
    record: dict = {"delegation_id": "d1"}
    for i in range(store.MAX_LIFECYCLE_EVENTS + 25):
        store.append_lifecycle_event(record, "heartbeat", f"i={i}")
    assert len(record["lifecycle"]) == store.MAX_LIFECYCLE_EVENTS
    last = record["lifecycle"][-1]
    assert last["detail"] == f"i={store.MAX_LIFECYCLE_EVENTS + 24}"


def test_first_cancel_wins_attribution_not_overwritten():
    record = _legacy_record()
    _write_record(record)
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    first = _load()["records"][record["delegation_id"]]["cancel_attribution"]
    # A second terminal path racing in must not rewrite history.
    store.cancel_matching(all_active=True, reason="/stop", caller="late-caller")
    second = _load()["records"][record["delegation_id"]]["cancel_attribution"]
    assert second == first
    assert second["reason"] == "session_end"


# --- additive schema / backward compatibility -------------------------------

def test_pre_upgrade_record_parses_and_gains_attribution_on_cancel():
    record = _legacy_record()
    _write_record(record)
    # Old record (no lifecycle, no cancel_attribution) loads without error.
    loaded = store.read_registry()["records"][record["delegation_id"]]
    assert "cancel_attribution" not in loaded
    assert "lifecycle" not in loaded
    assert store.cancel_matching(all_active=True, reason="/stop", caller="test") == 1
    stored = store.read_registry()["records"][record["delegation_id"]]
    assert stored["cancel_attribution"]["reason"] == "/stop"
    assert stored["cancel_attribution"]["caller"] == "test"
    assert [e["event"] for e in stored["lifecycle"]] == ["cancelled"]


def test_new_fields_survive_integrity_checksum_roundtrip():
    result, gate = _dispatch()
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    # read_registry re-validates every record checksum: the attributed record
    # must round-trip (integrity covers the new additive fields too).
    stored = store.read_registry()["records"][result["delegation_id"]]
    assert stored["cancel_attribution"]["reason"] == "session_end"
    gate.set()


def test_old_reader_contract_death_guard_fields_unchanged():
    """The delegation-death-guard reads state/terminal/outbox: all unchanged."""
    result, gate = _dispatch()
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    record = _load()["records"][result["delegation_id"]]
    # The guard's silent-death fingerprint fields keep their v1 shape.
    assert record["state"] == "cancelled"
    assert record["terminal"] is None
    assert isinstance(record["outbox"], list)
    # ...and now the record answers "who killed it" without a repro.
    assert record["cancel_attribution"]["reason"] == "session_end"
    gate.set()


def test_list_async_delegations_surfaces_attribution_summary():
    record = _legacy_record()
    _write_record(record)
    store.cancel_matching(all_active=True, reason="/stop", caller="cli.py:1:main")
    rows = ad.list_async_delegations()
    row = next(r for r in rows if r["delegation_id"] == record["delegation_id"])
    assert row["status"] == "cancelled"
    assert row["cancel_attribution"]["reason"] == "/stop"
    assert row["cancel_attribution"]["caller"] == "cli.py:1:main"
    assert "cancelled_at" in row["cancel_attribution"]


def test_attribution_stamp_never_breaks_cancel_on_weird_record():
    # A hand-edited record with a non-list lifecycle must still cancel cleanly.
    record = _legacy_record()
    record["lifecycle"] = "corrupted"
    _write_record(record)
    assert store.cancel_matching(all_active=True, reason="/stop", caller="t") == 1
    stored = _load()["records"][record["delegation_id"]]
    assert stored["state"] == "cancelled"
    assert stored["cancel_attribution"]["reason"] == "/stop"


def test_attribution_visible_while_in_memory_handle_still_running():
    # Greptile P2: a durably-cancelled record whose runner thread is still
    # alive appears in the in-memory snapshot; attribution must not be hidden
    # during that window.
    result, gate = _dispatch()
    ad.interrupt_for_session(parent_session_id="parent-1", reason="session_end")
    rows = ad.list_async_delegations()
    row = next(r for r in rows if r["delegation_id"] == result["delegation_id"])
    assert row["cancel_attribution"]["reason"] == "session_end"
    gate.set()


def test_transition_owned_cancel_records_delegation_ids_selector():
    record = _legacy_record()
    _write_record(record)
    assert store.transition_owned(
        [record["delegation_id"]], "cancelled", reason="forced", caller="t"
    )
    stored = _load()["records"][record["delegation_id"]]
    attribution = stored["cancel_attribution"]
    assert attribution["via"] == "transition_owned"
    assert attribution["selector"] == {"delegation_ids": [record["delegation_id"]]}
