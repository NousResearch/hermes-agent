"""Stall/crash parity with owner-death recovery (#116000): a synthetic terminal batch result must
not wipe the finished children ``record_unit_child`` already durably recorded on the unit's row.

``_stalled_result`` and the worker's crash handler both synthesized ``_batch_crash`` results
(``results: []``), so when the stale monitor killed a wedged sibling of a still-unfinished group
— or the unit runner raised after some children finished — ``_persist_completion`` overwrote
``result_json`` and the terminal event carried an empty results wall. The finished work recorded
for exactly this crash window was thrown away, while the owner-death path
(``recover_abandoned_delegations`` -> ``_recovered_results``) replays it.
"""

import json
import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_for(delegation_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def _record_finished_child(delegation_id: str) -> None:
    """Durably record one finished child on the unit's still-running row (what
    ``_run_children_parallel`` does via ``record_unit_child`` for a detached unit)."""
    ad.record_unit_child(delegation_id, {
        "task_index": 0, "status": "completed", "summary": "finished before the sibling wedged",
        "api_calls": 2, "duration_seconds": 1.5,
    })
    with ad._DB_LOCK, ad._transaction() as conn:
        row = conn.execute(
            "SELECT result_json FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
    assert row is not None, "dispatch must persist the unit row before returning"
    assert [r["summary"] for r in json.loads(row[0])["results"]] == ["finished before the sibling wedged"]


def test_stalled_batch_keeps_recorded_children(monkeypatch):
    """A group unit where one child finished (recorded durably) and a sibling wedges: the forced
    ``stalled`` completion must carry the finished child's real result, not an empty results list."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    # Long idle while the finished child is recorded, then tightened to trip the stall.
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 30.0)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", 30.0)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.05)
    gate = threading.Event()

    def runner():
        # The unit runner never returns (a child is wedged in an uninterruptible call).
        gate.wait(timeout=30)
        return {"results": [], "total_duration_seconds": 0}

    res = ad.dispatch_async_delegation_batch(
        goals=["fast child", "wedged child"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1, delegation_id="deleg_stallrec",
        # Frozen progress token for both children: the batch looks wedged.
        progress_fn=lambda: (((0, None, None), (0, None, None)), False),
    )
    assert res["status"] == "dispatched"
    _record_finished_child("deleg_stallrec")

    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", 0.1)
    evt = _drain_for("deleg_stallrec")
    try:
        assert evt is not None
        assert evt["status"] == "stalled"
        assert evt["is_batch"] is True
        by_index = {r.get("task_index"): r for r in evt.get("results") or []}
        # The finished child's real result survives the force-finalization.
        assert by_index.get(0, {}).get("summary") == "finished before the sibling wedged"
        # The wedged sibling is honestly reported, not silently dropped.
        assert by_index.get(1, {}).get("status") == "unknown"
        assert evt.get("error")
        # The model-facing wall shows the recovered work (same rendering as the owner-death path).
        assert "finished before the sibling wedged" in (format_process_notification(evt) or "")
    finally:
        gate.set()

    # The durable row keeps the recorded result too (restart replay reads result_json).
    durable = ad.get_durable_delegation("deleg_stallrec")
    assert durable is not None and durable["delivery_state"] in ("pending", "delivered")
    recorded_summaries = [r.get("summary") for r in (durable["result"] or {}).get("results") or []]
    assert "finished before the sibling wedged" in recorded_summaries


def test_crashed_batch_keeps_recorded_children():
    """A unit runner that RAISES after some children finished (hook/transcript finalize error) must
    not lose the durably recorded children either — same class as the stall path."""
    gate = threading.Event()

    def runner():
        gate.wait(timeout=10)  # the child finished and was recorded first
        raise RuntimeError("post-join finalize exploded")

    res = ad.dispatch_async_delegation_batch(
        goals=["fast child", "doomed child"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1, delegation_id="deleg_crashrec",
    )
    assert res["status"] == "dispatched"
    _record_finished_child("deleg_crashrec")
    gate.set()

    evt = _drain_for("deleg_crashrec")
    assert evt is not None
    assert evt["status"] == "error"
    by_index = {r.get("task_index"): r for r in evt.get("results") or []}
    assert by_index.get(0, {}).get("summary") == "finished before the sibling wedged"
    assert by_index.get(1, {}).get("status") == "unknown"
    assert "post-join finalize exploded" in evt.get("error", "")


def test_stalled_batch_without_recorded_children_keeps_empty_results(monkeypatch):
    """No recorded children (e.g. a runner crash before any child finished): the synthetic terminal
    event still reports the batch error with no per-task results — the recovery merge must not
    invent results."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", 0.1)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.05)
    gate = threading.Event()

    def runner():
        gate.wait(timeout=30)
        return {"results": [], "total_duration_seconds": 0}

    res = ad.dispatch_async_delegation_batch(
        goals=["only child"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1, delegation_id="deleg_stallempty",
        progress_fn=lambda: (((0, None, None),), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for("deleg_stallempty")
    try:
        assert evt is not None
        assert evt["status"] == "stalled"
        assert evt.get("error")
        # No fabricated per-task results for a unit that recorded none.
        assert evt.get("results") in (None, [], [])
    finally:
        gate.set()
