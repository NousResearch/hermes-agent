"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-released workers a beat to finalize BEFORE draining, so their
    # completion events land now instead of leaking into the next test's
    # queue (worker threads push events asynchronously; a drain that races an
    # in-flight _finalize misses it).
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _drain_for(delegation_id, timeout=5.0):
    """Drain until the event for *delegation_id* appears (discarding others).

    Completion events are pushed asynchronously by worker threads, so a
    straggler from a PREVIOUS test can land after that test's teardown drain
    and leak into the current test's queue. Matching on delegation_id makes
    the assertion immune to that cross-test leak.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def test_active_for_session_counts_every_live_delegation_state():
    with ad._records_lock:
        ad._records.update(
            {
                "queued": {
                    "delegation_id": "queued",
                    "status": "queued",
                    "origin_ui_session_id": "desktop-sid",
                },
                "running": {
                    "status": "running",
                    "origin_ui_session_id": "desktop-sid",
                },
                "stalling": {
                    "status": "stalling",
                    "origin_ui_session_id": "desktop-sid",
                },
                "finalizing": {
                    "status": "finalizing",
                    "origin_ui_session_id": "desktop-sid",
                },
                "completed": {
                    "status": "completed",
                    "origin_ui_session_id": "desktop-sid",
                },
                "other-session": {
                    "status": "running",
                    "origin_ui_session_id": "other-sid",
                },
            }
        )

    assert ad.active_for_session("desktop-sid") == 4
    assert ad.active_for_session("other-sid") == 1
    assert ad.active_for_session("") == 0
    assert ad.has_live_for_session(origin_ui_session_id="desktop-sid") is True
    with ad._records_lock:
        ad._records.pop("queued", None)


def test_dispatch_returns_immediately_without_blocking():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m"}

    t0 = time.monotonic()
    res = ad.dispatch_async_delegation(
        goal="g", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=3,
    )
    elapsed = time.monotonic() - t0

    assert res["status"] == "dispatched"
    assert res["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: dispatch returned while the runner is still
    # gated (active), so it cannot have waited on the gate. The active_count
    # check is the environment-independent proof; the generous wall-clock
    # bound is a loose sanity backstop, not the primary assertion (a loaded
    # CI runner can be slow but never anywhere near the runner's 5s gate).
    assert ad.active_count() == 1
    assert elapsed < 4.0, f"dispatch blocked {elapsed:.2f}s (gate is 5s)"
    gate.set()


def test_async_executor_workers_are_daemon_threads():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done"}

    res = ad.dispatch_async_delegation(
        goal="daemon check", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert res["status"] == "dispatched"

    deadline = time.monotonic() + 2
    worker = None
    while time.monotonic() < deadline:
        worker = next(
            (t for t in threading.enumerate() if t.name.startswith("async-delegate")),
            None,
        )
        if worker is not None:
            break
        time.sleep(0.02)
    assert worker is not None
    assert worker.daemon is True
    gate.set()
    assert _drain_one() is not None


def test_completion_event_lands_on_shared_queue_with_session_key():
    def runner():
        return {"status": "completed", "summary": "the result",
                "api_calls": 3, "duration_seconds": 2.0, "model": "test-model"}

    res = ad.dispatch_async_delegation(
        goal="compute X", context="some context", toolsets=["web", "file"],
        role="leaf", model="test-model", session_key="agent:main:cli:dm:local",
        parent_session_id="20260703_parent_sid",
        runner=runner, max_async_children=3,
    )
    assert res["status"] == "dispatched"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["summary"] == "the result"
    assert evt["session_key"] == "agent:main:cli:dm:local"
    assert evt["parent_session_id"] == "20260703_parent_sid"
    assert evt["delegation_id"] == res["delegation_id"]


def test_rich_reinjection_block_is_self_contained():
    def runner():
        return {"status": "completed", "summary": "The answer is 42.",
                "api_calls": 7, "duration_seconds": 3.5, "model": "test-model"}

    ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"], role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=3,
    )
    evt = _drain_one()
    assert evt is not None
    text = format_process_notification(evt)
    assert text is not None
    for needle in [
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ]:
        assert needle in text, f"missing {needle!r}"


def test_dispatch_queues_at_capacity_and_starts_after_slot_frees():
    first_gate = threading.Event()
    second_started = threading.Event()

    def first_runner():
        first_gate.wait(timeout=60)
        return {"status": "completed", "summary": "first"}

    def second_runner():
        second_started.set()
        return {"status": "completed", "summary": "second"}

    first = ad.dispatch_async_delegation(
        goal="first", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=first_runner, max_async_children=1,
    )
    assert first["status"] == "dispatched"

    second = ad.dispatch_async_delegation(
        goal="second", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=second_runner, max_async_children=1,
    )
    assert second["status"] == "queued"
    assert second["delegation_id"].startswith("deleg_")
    assert second_started.wait(timeout=0.1) is False

    first_gate.set()
    assert _drain_for(first["delegation_id"]) is not None
    assert second_started.wait(timeout=2)
    assert _drain_for(second["delegation_id"]) is not None


def test_batch_queues_until_all_child_slots_are_available():
    first_gate = threading.Event()
    batch_started = threading.Event()

    def first_runner():
        first_gate.wait(timeout=60)
        return {"status": "completed", "summary": "first"}

    def batch_runner():
        batch_started.set()
        return {
            "results": [
                {"status": "completed", "summary": "a"},
                {"status": "completed", "summary": "b"},
            ]
        }

    first = ad.dispatch_async_delegation(
        goal="first", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=first_runner, max_async_children=2,
    )
    assert first["status"] == "dispatched"

    batch = ad.dispatch_async_delegation_batch(
        goals=["a", "b"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=batch_runner, max_async_children=2,
    )
    assert batch["status"] == "queued"
    assert batch_started.wait(timeout=0.1) is False
    assert ad.active_task_count() == 1

    first_gate.set()
    assert _drain_for(first["delegation_id"]) is not None
    assert batch_started.wait(timeout=2)
    assert _drain_for(batch["delegation_id"]) is not None


def test_pending_queue_is_bounded_and_overflow_never_runs():
    first_gate = threading.Event()
    overflow_started = threading.Event()

    def blocker():
        first_gate.wait(timeout=60)
        return {"status": "completed", "summary": "done"}

    first = ad.dispatch_async_delegation(
        goal="first", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=1,
        max_queued_delegations=1,
    )
    queued = ad.dispatch_async_delegation(
        goal="queued", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=1,
        max_queued_delegations=1,
    )
    overflow = ad.dispatch_async_delegation(
        goal="overflow", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: overflow_started.set() or {},
        max_async_children=1, max_queued_delegations=1,
    )

    assert first["status"] == "dispatched"
    assert queued["status"] == "queued"
    assert overflow["status"] == "rejected"
    assert "queue" in overflow["error"].lower()
    assert overflow_started.wait(timeout=0.1) is False

    first_gate.set()
    assert _drain_for(first["delegation_id"]) is not None
    assert _drain_for(queued["delegation_id"]) is not None


def test_queue_is_fifo_when_head_batch_needs_more_slots():
    first_gate = threading.Event()
    order = []

    def active_runner():
        first_gate.wait(timeout=60)
        return {"status": "completed", "summary": "active"}

    first = ad.dispatch_async_delegation(
        goal="active", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=active_runner,
        max_async_children=3,
    )
    batch = ad.dispatch_async_delegation_batch(
        goals=["a", "b", "c"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: order.append("batch") or {"results": []},
        max_async_children=3,
    )
    later = ad.dispatch_async_delegation(
        goal="later", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: order.append("later") or {
            "status": "completed", "summary": "later"
        },
        max_async_children=3,
    )

    assert first["status"] == "dispatched"
    assert batch["status"] == "queued"
    assert later["status"] == "queued"
    assert order == []

    first_gate.set()
    assert _drain_for(first["delegation_id"]) is not None
    assert _drain_for(batch["delegation_id"]) is not None
    assert _drain_for(later["delegation_id"]) is not None
    assert order == ["batch", "later"]


def test_dispatch_waits_for_memory_floor_before_starting(monkeypatch):
    available = {"bytes": 100}
    started = threading.Event()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(
        ad, "_effective_available_memory_bytes", lambda: available["bytes"]
    )

    result = ad.dispatch_async_delegation(
        goal="memory gated", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {
            "status": "completed", "summary": "done"
        },
        max_async_children=1,
        min_available_memory_bytes=200,
    )

    assert result["status"] == "queued"
    assert result["queue_reason"] == "resources"
    assert started.wait(timeout=0.1) is False

    available["bytes"] = 300
    assert started.wait(timeout=1)
    assert _drain_for(result["delegation_id"]) is not None


def test_effective_memory_uses_tightest_cgroup_limit(monkeypatch, tmp_path):
    import psutil

    cgroup = tmp_path / "delegation.slice"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("100\n", encoding="utf-8")
    (cgroup / "memory.high").write_text("700\n", encoding="utf-8")
    (cgroup / "memory.max").write_text("900\n", encoding="utf-8")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1000)
    )
    monkeypatch.setattr(ad, "_current_cgroup_v2_path", lambda: str(cgroup))

    assert ad._effective_available_memory_bytes() == 600


def test_memory_probe_failure_fails_closed_when_floor_configured(monkeypatch):
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: None)
    record = {
        "min_available_memory_bytes": 100,
        "resume_available_memory_bytes": 100,
        "max_memory_psi_avg10": 0.0,
        "resume_memory_psi_avg10": 0.0,
        "_resource_blocked": False,
    }
    assert ad._resources_available(record) is False
    assert record["_resource_blocked"] is True


def test_memory_probe_failure_without_floor_allows_admission(monkeypatch):
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: None)
    record = {
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
        "max_memory_psi_avg10": 0.0,
        "resume_memory_psi_avg10": 0.0,
        "_resource_blocked": False,
    }
    assert ad._resources_available(record) is True


def test_psi_unavailable_falls_back_to_memory_gate(monkeypatch):
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 500)
    monkeypatch.setattr(ad, "_memory_psi_avg10", lambda: None)
    record = {
        "min_available_memory_bytes": 100,
        "resume_available_memory_bytes": 100,
        "max_memory_psi_avg10": 50.0,
        "resume_memory_psi_avg10": 50.0,
        "_resource_blocked": False,
    }
    # PSI unavailable is NOT a block; memory gate decides.
    assert ad._resources_available(record) is True


def test_psi_over_ceiling_blocks_until_recovery(monkeypatch):
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 500)
    pressure = {"avg10": 70.0}
    monkeypatch.setattr(ad, "_memory_psi_avg10", lambda: pressure["avg10"])
    record = {
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
        "max_memory_psi_avg10": 50.0,
        "resume_memory_psi_avg10": 10.0,
        "_resource_blocked": False,
    }
    assert ad._resources_available(record) is False
    # Still over the stop ceiling.
    pressure["avg10"] = 60.0
    assert ad._resources_available(record) is False
    # Dropped below the resume ceiling -> admitted again.
    pressure["avg10"] = 5.0
    assert ad._resources_available(record) is True


def test_non_posix_returns_host_availability_only(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    # On non-POSIX the cgroup probe is skipped entirely.
    monkeypatch.setattr(ad, "_current_cgroup_v2_path", lambda: None)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1234))
    assert ad._effective_available_memory_bytes() == 1234
    assert ad._memory_psi_avg10() is None
    monkeypatch.setattr(os, "name", "posix")


def test_missing_cgroup_falls_back_to_host_availability(monkeypatch):
    monkeypatch.setattr(ad, "_current_cgroup_v2_path", lambda: None)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=777))
    assert ad._effective_available_memory_bytes() == 777


def test_memory_hysteresis_requires_resume_floor(monkeypatch):
    available = {"bytes": 50}
    started = threading.Event()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(
        ad, "_effective_available_memory_bytes", lambda: available["bytes"]
    )

    result = ad.dispatch_async_delegation(
        goal="memory hysteresis", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1,
        min_available_memory_bytes=100,
        resume_available_memory_bytes=200,
    )
    assert result["status"] == "queued"

    available["bytes"] = 150
    assert started.wait(timeout=0.1) is False
    available["bytes"] = 200
    assert started.wait(timeout=1)
    assert _drain_for(result["delegation_id"]) is not None


def test_memory_psi_gate_uses_lower_resume_threshold(monkeypatch):
    pressure = {"avg10": 20.0}
    started = threading.Event()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 10_000)
    monkeypatch.setattr(ad, "_memory_psi_avg10", lambda: pressure["avg10"])

    result = ad.dispatch_async_delegation(
        goal="psi gated", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1,
        max_memory_psi_avg10=10.0,
        resume_memory_psi_avg10=5.0,
    )
    assert result["status"] == "queued"

    pressure["avg10"] = 8.0
    assert started.wait(timeout=0.1) is False
    pressure["avg10"] = 4.0
    assert started.wait(timeout=1)
    assert _drain_for(result["delegation_id"]) is not None


def test_batch_larger_than_slot_limit_is_rejected_without_runner():
    started = threading.Event()
    result = ad.dispatch_async_delegation_batch(
        goals=["a", "b"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {"results": []},
        max_async_children=1,
    )
    assert result["status"] == "rejected"
    assert "requires 2 child slots" in result["error"]
    assert started.is_set() is False


def test_dispatch_persistence_failure_releases_reservation(monkeypatch):
    started = threading.Event()

    def fail_persistence(_record):
        raise OSError("state db unavailable")

    monkeypatch.setattr(ad, "_persist_dispatch", fail_persistence)
    result = ad.dispatch_async_delegation(
        goal="must persist", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1,
    )

    assert result["status"] == "rejected"
    assert "persist" in result["error"].lower()
    assert ad.active_count() == 0
    assert ad.list_async_delegations() == []
    assert started.is_set() is False


def test_concurrent_queue_overflow_never_exceeds_bound():
    active_started = threading.Event()
    release_active = threading.Event()

    first = ad.dispatch_async_delegation(
        goal="active", context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=1, max_queued_delegations=3,
        runner=lambda: (
            active_started.set(), release_active.wait(timeout=2),
            {"status": "completed", "summary": "active"},
        )[-1],
    )
    assert first["status"] == "dispatched"
    assert active_started.wait(timeout=1)

    barrier = threading.Barrier(11)
    results = []
    result_lock = threading.Lock()

    def submit(index):
        barrier.wait()
        outcome = ad.dispatch_async_delegation(
            goal=f"queued-{index}", context=None, toolsets=None, role="leaf", model="m",
            session_key="", max_async_children=1, max_queued_delegations=3,
            runner=lambda: {"status": "completed", "summary": str(index)},
        )
        with result_lock:
            results.append(outcome)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert sum(result["status"] == "queued" for result in results) == 3
    assert sum(result["status"] == "rejected" for result in results) == 7
    assert sum(
        record["status"] == "queued" for record in ad.list_async_delegations()
    ) == 3

    release_active.set()
    completion_ids = {first["delegation_id"]}
    completion_ids.update(
        result["delegation_id"] for result in results if result["status"] == "queued"
    )
    seen = set()
    deadline = time.monotonic() + 3
    while completion_ids - seen and time.monotonic() < deadline:
        try:
            event = process_registry.completion_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        seen.add(event.get("delegation_id"))
    assert completion_ids <= seen


def test_interrupt_all_cancels_queued_work_without_starting_it(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)

    result = ad.dispatch_async_delegation(
        goal="queued cancel", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert result["status"] == "queued"

    assert ad.interrupt_all(reason="test shutdown") == 1
    event = _drain_for(result["delegation_id"])
    assert event is not None
    assert event["status"] == "interrupted"
    assert started.wait(timeout=0.1) is False

    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 100)
    time.sleep(0.1)
    assert started.is_set() is False


def test_queued_work_times_out_without_starting(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.01)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)

    result = ad.dispatch_async_delegation(
        goal="queued timeout", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1, min_available_memory_bytes=1,
        queue_timeout_seconds=0.05,
    )
    assert result["status"] == "queued"

    event = _drain_for(result["delegation_id"], timeout=2)
    assert event is not None
    assert event["status"] == "timeout"
    assert "queue" in event["error"].lower()
    assert started.is_set() is False


def test_capacity_queued_timeout_fires_while_worker_slot_is_busy():
    gate = threading.Event()
    queued_started = threading.Event()

    def busy_runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "busy"}

    first = ad.dispatch_async_delegation(
        goal="busy", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=busy_runner,
        max_async_children=1,
    )
    queued = ad.dispatch_async_delegation(
        goal="capacity timeout", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: queued_started.set() or {},
        max_async_children=1, queue_timeout_seconds=0.05,
    )
    assert first["status"] == "dispatched"
    assert queued["status"] == "queued"

    event = _drain_for(queued["delegation_id"], timeout=1)
    assert event is not None
    assert event["status"] == "timeout"
    assert queued_started.is_set() is False

    gate.set()
    assert _drain_for(first["delegation_id"]) is not None


def test_lingering_timed_out_child_retains_slot_until_future_exits():
    lingering = Future()
    next_started = threading.Event()

    timed_out = ad.dispatch_async_delegation(
        goal="timed out but alive", context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=1,
        runner=lambda: {
            "status": "error",
            "summary": None,
            "error": "timed out",
            "_lingering_futures": [lingering],
        },
    )
    assert timed_out["status"] == "dispatched"
    event = _drain_for(timed_out["delegation_id"])
    assert event is not None
    assert event["status"] == "error"

    queued = ad.dispatch_async_delegation(
        goal="must wait for actual exit", context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=1,
        runner=lambda: next_started.set() or {"status": "completed"},
    )
    assert queued["status"] == "queued"
    assert next_started.wait(timeout=0.1) is False

    lingering.set_result(None)
    assert next_started.wait(timeout=1)
    assert _drain_for(queued["delegation_id"]) is not None


def test_batch_lingering_futures_retain_slots_until_they_exit():
    lingering_a = Future()
    next_started = threading.Event()

    timed_out_batch = ad.dispatch_async_delegation_batch(
        goals=["a", "b"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=2,
        runner=lambda: {
            "status": "completed",
            "results": [
                {"task_index": 0, "status": "timeout", "summary": None},
                {"task_index": 1, "status": "completed", "summary": "ok"},
            ],
            "_lingering_futures": [lingering_a],
        },
    )
    assert timed_out_batch["status"] == "dispatched"
    event = _drain_for(timed_out_batch["delegation_id"])
    assert event is not None
    assert event["status"] == "completed"

    queued = ad.dispatch_async_delegation_batch(
        goals=["c", "d"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=2,
        runner=lambda: next_started.set() or {"status": "completed", "results": []},
    )
    assert queued["status"] == "queued"
    assert next_started.wait(timeout=0.1) is False

    lingering_a.set_result(None)
    assert next_started.wait(timeout=1)
    assert _drain_for(queued["delegation_id"]) is not None


def test_reset_for_tests_clears_lingering_slots():
    lingering = Future()
    result = ad.dispatch_async_delegation(
        goal="reset clears slots", context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=1,
        runner=lambda: {
            "status": "error", "summary": None, "error": "timed out",
            "_lingering_futures": [lingering],
        },
    )
    assert result["status"] == "dispatched"
    assert _drain_for(result["delegation_id"]) is not None
    assert sum(ad._lingering_resource_slots.values()) >= 1

    ad._reset_for_tests()
    assert ad._lingering_resource_slots == {}


def test_stale_generation_worker_cannot_push_completion_after_reset():
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    stale = {
        "delegation_id": "deleg_stale_generation",
        "session_key": "session-stale",
        "status": "finalizing",
        "dispatched_at": time.time(),
        "completed_at": time.time(),
        "goal": "stale generation",
        # Captured before the reset that bumped the generation.
        "_generation": ad._manager_generation - 1,
    }
    with ad._records_lock:
        ad._records[stale["delegation_id"]] = stale

    ad._push_completion_event(
        stale, {"status": "completed", "summary": "must not leak"}, "completed"
    )

    assert process_registry.completion_queue.empty()
    assert ad.get_durable_delegation("deleg_stale_generation") is None
    ad._reset_for_tests()


def test_late_worker_after_reset_does_not_leak_completion():
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    started = threading.Event()
    release = threading.Event()

    result = ad.dispatch_async_delegation(
        goal="late worker", context=None, toolsets=None, role="leaf", model="m",
        session_key="", max_async_children=1,
        runner=lambda: (
            started.set(), release.wait(timeout=5),
            {"status": "completed", "summary": "late"},
        )[-1],
    )
    assert result["status"] == "dispatched"
    assert started.wait(timeout=1)

    ad._reset_for_tests()
    release.set()
    time.sleep(0.3)

    assert process_registry.completion_queue.empty()
    assert ad.list_async_delegations() == []


def test_queued_interruption_falls_back_to_interrupt_fn_after_promotion():
    """C1: interrupt racing promotion must not terminalize a promoted record."""
    ad._reset_for_tests()
    interrupt_calls = []

    def interrupt_fn():
        interrupt_calls.append(True)

    record = {
        "delegation_id": "deleg_promote_race",
        "session_key": "session-c1",
        "status": "queued",
        "dispatched_at": time.time(),
        "completed_at": None,
        "goal": "promotion race",
        "is_batch": False,
        "interrupt_fn": interrupt_fn,
        "progress_fn": None,
        "required_slots": 1,
        "max_async_children": 1,
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
        "max_memory_psi_avg10": 0.0,
        "resume_memory_psi_avg10": 0.0,
        "queue_timeout_seconds": 0.0,
        "_queued_monotonic": time.monotonic(),
        "_progress_token": None,
        "_progress_ts": time.time(),
        "_interrupted_at": None,
        "_generation": ad._manager_generation,
    }
    with ad._records_lock:
        ad._records[record["delegation_id"]] = record

    # Admission thread wins the race: record is promoted to running before
    # the interrupt path claims it.
    with ad._records_lock:
        ad._records[record["delegation_id"]]["status"] = "running"

    result = ad._finalize_queued_interruption(record["delegation_id"], "race test")
    assert result is True
    assert interrupt_calls == [True]
    # Record must NOT be terminalized by the queued path.
    assert ad._records[record["delegation_id"]]["status"] == "running"
    from tools.process_registry import process_registry

    assert process_registry.completion_queue.empty()
    ad._reset_for_tests()


def test_queued_interruption_after_promotion_delivers_one_interrupted(monkeypatch):
    """End-to-end C1: interrupt during admission -> exactly one interrupted event."""
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)
    started = threading.Event()
    gate = threading.Event()

    def runner():
        started.set()
        gate.wait(timeout=2)
        return {"status": "completed", "summary": "should not complete"}

    # Capacity-freeze the slot so the second dispatch queues.
    blocker = ad.dispatch_async_delegation(
        goal="blocker", context=None, toolsets=None, role="leaf", model="m",
        session_key="c1-e2e", runner=lambda: (
            gate.wait(timeout=2), {"status": "completed"},
        )[-1],
        max_async_children=1, min_available_memory_bytes=0,
    )
    assert blocker["status"] == "dispatched"
    # Queue a second one behind it, then interrupt the session immediately.
    queued = ad.dispatch_async_delegation(
        goal="raced", context=None, toolsets=None, role="leaf", model="m",
        session_key="c1-e2e", runner=runner,
        max_async_children=1, min_available_memory_bytes=0,
    )
    assert queued["status"] == "queued"
    assert ad.interrupt_for_session(session_key="c1-e2e", reason="race") >= 1

    gate.set()  # let the blocker finish so the queue drains
    deadline = time.monotonic() + 3
    events = []
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            events.append(process_registry.completion_queue.get_nowait())
        time.sleep(0.02)
    ids = [e.get("delegation_id") for e in events]
    # Both delegations must resolve exactly once; the raced one must be
    # interrupted (via interrupt_fn fallback), never completed.
    assert blocker["delegation_id"] in ids
    raced_events = [e for e in events if e.get("delegation_id") == queued["delegation_id"]]
    assert len(raced_events) == 1
    assert raced_events[0]["status"] == "interrupted"
    ad._reset_for_tests()


def test_persist_state_failure_during_admission_still_submits_worker(monkeypatch):
    """M2: a _persist_state failure must not kill the admission thread."""
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    available = {"bytes": 0}
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: available["bytes"])
    real_persist = ad._persist_state
    fail_next = {"fail": True}

    def flaky_persist(delegation_id, state):
        if fail_next["fail"] and state == "running":
            fail_next["fail"] = False
            raise OSError("simulated disk full")
        return real_persist(delegation_id, state)

    monkeypatch.setattr(ad, "_persist_state", flaky_persist)
    started = threading.Event()

    result = ad.dispatch_async_delegation(
        goal="persist-fail", context=None, toolsets=None, role="leaf", model="m",
        session_key="m2", runner=lambda: started.set() or {"status": "completed"},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert result["status"] == "queued"
    available["bytes"] = 100  # memory recovers -> admission proceeds
    event = _drain_for(result["delegation_id"])
    assert event is not None
    assert event["status"] == "completed"
    assert started.is_set()
    ad._reset_for_tests()


def test_stale_monitor_restarts_when_queued_record_admitted(monkeypatch):
    """M1: admission must re-ensure the stale monitor if it exited."""
    ad._reset_for_tests()
    calls = {"ensure": 0}
    real_ensure = ad._ensure_stale_monitor

    def counting_ensure():
        calls["ensure"] += 1
        return real_ensure()

    monkeypatch.setattr(ad, "_ensure_stale_monitor", counting_ensure)
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    available = {"bytes": 0}
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: available["bytes"])
    before = calls["ensure"]

    result = ad.dispatch_async_delegation(
        goal="m1", context=None, toolsets=None, role="leaf", model="m",
        session_key="m1", runner=lambda: {"status": "completed"},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert result["status"] == "queued"
    available["bytes"] = 100  # memory recovers -> admission proceeds
    event = _drain_for(result["delegation_id"])
    assert event is not None
    # Admission path must have re-ensured the monitor even though dispatch
    # was blocked (no progress_fn -> monitor exited during queued period).
    assert calls["ensure"] > before
    ad._reset_for_tests()


def test_queued_persist_failure_releases_slot_and_does_not_abort_sweep(monkeypatch):
    """MAJOR: a persist failure in queued finalization must not strand it."""
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)
    real_push = ad._push_completion_event
    fail_next = {"fail": True}

    def flaky_push(record, result, status):
        if fail_next["fail"]:
            fail_next["fail"] = False
            raise OSError("simulated DB full")
        return real_push(record, result, status)

    monkeypatch.setattr(ad, "_push_completion_event", flaky_push)

    # Two memory-gated queued records in different sessions. Strict FIFO
    # means the second queues behind the first; both are never-started.
    first = ad.dispatch_async_delegation(
        goal="maj-first", context=None, toolsets=None, role="leaf", model="m",
        session_key="maj1", runner=lambda: {"status": "completed"},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert first["status"] == "queued"
    second = ad.dispatch_async_delegation(
        goal="maj-second", context=None, toolsets=None, role="leaf", model="m",
        session_key="maj2", runner=lambda: {"status": "completed"},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert second["status"] == "queued"

    # interrupt_all must NOT abort mid-sweep when the first record's
    # completion persist raises: both records get a terminal state, and no
    # exception escapes the sweep.
    count = ad.interrupt_all(reason="maj test")
    assert count == 2
    with ad._records_lock:
        assert ad._records[first["delegation_id"]]["status"] == "interrupted"
        assert ad._records[second["delegation_id"]]["status"] == "interrupted"
    # Exactly one event delivered (the failing record's push was skipped;
    # the second record's succeeded).
    delivered = []
    while not process_registry.completion_queue.empty():
        delivered.append(process_registry.completion_queue.get_nowait())
    assert len(delivered) == 1
    assert delivered[0]["delegation_id"] == second["delegation_id"]
    # The slot is released: a fresh dispatch with memory available starts.
    available = {"bytes": 100}
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: available["bytes"])
    fresh = ad.dispatch_async_delegation(
        goal="maj-fresh", context=None, toolsets=None, role="leaf", model="m",
        session_key="maj3", runner=lambda: {"status": "completed"},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert fresh["status"] == "dispatched"
    ad._reset_for_tests()


def test_queued_interruption_skips_terminal_record(monkeypatch):
    """MINOR: fallback must not over-report interrupted for completed work."""
    ad._reset_for_tests()
    interrupt_calls = []

    def interrupt_fn():
        interrupt_calls.append(True)

    record = {
        "delegation_id": "deleg_terminal_skip",
        "session_key": "session-minor1",
        "status": "completed",  # worker already won the race and finished
        "dispatched_at": time.time(),
        "completed_at": time.time(),
        "goal": "terminal skip",
        "is_batch": False,
        "interrupt_fn": interrupt_fn,
        "progress_fn": None,
        "required_slots": 1,
        "max_async_children": 1,
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
        "max_memory_psi_avg10": 0.0,
        "resume_memory_psi_avg10": 0.0,
        "queue_timeout_seconds": 0.0,
        "_queued_monotonic": time.monotonic(),
        "_progress_token": None,
        "_progress_ts": time.time(),
        "_interrupted_at": None,
        "_generation": ad._manager_generation,
    }
    with ad._records_lock:
        ad._records[record["delegation_id"]] = record

    result = ad._finalize_queued_interruption(record["delegation_id"], "skip test")
    assert result is False
    assert interrupt_calls == []
    ad._reset_for_tests()


def test_stalled_finalize_persist_failure_still_releases_slot(monkeypatch):
    """Cycle-3 MAJOR: _finalize_stalled must not strand on persist failure."""
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    real_push = ad._push_completion_event

    def flaky_push(record, result, status):
        raise OSError("simulated DB full")

    monkeypatch.setattr(ad, "_push_completion_event", flaky_push)
    record = {
        "delegation_id": "deleg_stalled_persist",
        "session_key": "c3-major",
        "status": "stalling",
        "dispatched_at": time.time() - 60,
        "completed_at": None,
        "goal": "stalled persist",
        "is_batch": False,
        "interrupt_fn": None,
        "progress_fn": None,
        "required_slots": 1,
        "max_async_children": 1,
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
        "max_memory_psi_avg10": 0.0,
        "resume_memory_psi_avg10": 0.0,
        "queue_timeout_seconds": 0.0,
        "_queued_monotonic": time.monotonic(),
        "_progress_token": None,
        "_progress_ts": time.time(),
        "_interrupted_at": time.time(),
        "_stall_quiet_seconds": 5.0,
        "_stall_threshold_seconds": 5.0,
        "_stall_in_tool": False,
        "_generation": ad._manager_generation,
    }
    with ad._records_lock:
        ad._records[record["delegation_id"]] = record

    ad._finalize_stalled(record["delegation_id"])
    # Record must be terminalized (slot released) despite the persist failure.
    with ad._records_lock:
        assert ad._records[record["delegation_id"]]["status"] == "stalled"
    assert process_registry.completion_queue.empty()
    ad._reset_for_tests()


def test_psi_resume_zero_follows_stop_ceiling():
    """Cycle-3 MINOR: resume PSI 0 must mean 'follow stop ceiling'."""
    ad._reset_for_tests()
    record = {
        "delegation_id": "deleg_psi_normalize",
        "session_key": "c3-minor2",
        "status": "queued",
        "max_memory_psi_avg10": 50.0,
        "resume_memory_psi_avg10": 0.0,  # direct caller passes 0 explicitly
        "min_available_memory_bytes": 0,
        "resume_available_memory_bytes": 0,
    }
    # Not blocked: stop ceiling (50) applies.
    assert ad._resources_available(record) is True
    # Blocked once: resume ceiling must follow stop ceiling (50), not clamp to 0.
    record["_resource_blocked"] = True
    assert ad._resources_available(record) is True
    ad._reset_for_tests()


def test_prune_durable_records_preserves_live_queued_rows(monkeypatch, tmp_path):
    """M3: durable pruning must never delete a live queued row."""
    import gateway.status as gateway_status

    ad._db_path = lambda: tmp_path / "state.db"
    monkeypatch.setattr(gateway_status, "_pid_exists", lambda _pid: False)
    now = time.time()
    # A live queued row with a frozen updated_at (oldest pending row).
    ad._persist_dispatch({
        "delegation_id": "deleg_live_queued",
        "session_key": "m3",
        "status": "queued",
        "dispatched_at": now - 1000,
        "goal": "live queued",
        "is_batch": False,
    })
    # A terminal delivered row (candidate for pruning).
    ad._persist_dispatch({
        "delegation_id": "deleg_terminal_1",
        "session_key": "m3",
        "status": "completed",
        "dispatched_at": now - 1000,
        "goal": "terminal",
        "is_batch": False,
    })
    ad._persist_completion(
        {"delegation_id": "deleg_terminal_1", "session_key": "m3",
         "status": "completed", "dispatched_at": now - 1000,
         "completed_at": now - 500},
        {"status": "completed"},
    )
    ad.mark_completion_delivered("deleg_terminal_1")
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET updated_at=? WHERE delegation_id='deleg_terminal_1'",
            (now - 500,),
        )

    ad._prune_durable_records()

    durable = ad.get_durable_delegation("deleg_live_queued")
    assert durable is not None
    assert durable["state"] == "queued"


def test_interrupt_for_session_cancels_only_matching_queued_work(monkeypatch):
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)

    target = ad.dispatch_async_delegation(
        goal="target", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-a", runner=lambda: {}, max_async_children=1,
        min_available_memory_bytes=1,
    )
    other = ad.dispatch_async_delegation(
        goal="other", context=None, toolsets=None, role="leaf", model="m",
        session_key="session-b", runner=lambda: {}, max_async_children=1,
        min_available_memory_bytes=1,
    )

    assert ad.interrupt_for_session(session_key="session-a") == 1
    event = _drain_for(target["delegation_id"])
    assert event is not None
    assert event["status"] == "interrupted"
    other_items = [
        item for item in ad.list_async_delegations()
        if item["delegation_id"] == other["delegation_id"]
    ]
    assert other_items[0]["status"] == "queued"

    assert ad.interrupt_all(reason="test cleanup") == 1
    assert _drain_for(other["delegation_id"]) is not None


def test_begin_shutdown_cancels_queue_and_rejects_new_work(monkeypatch):
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(ad, "_effective_available_memory_bytes", lambda: 0)
    started = threading.Event()

    queued = ad.dispatch_async_delegation(
        goal="queued at shutdown", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1, min_available_memory_bytes=1,
    )
    assert queued["status"] == "queued"

    assert ad.begin_shutdown(reason="test shutdown") == 1
    event = _drain_for(queued["delegation_id"])
    assert event is not None
    assert event["status"] == "interrupted"

    rejected = ad.dispatch_async_delegation(
        goal="too late", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=lambda: started.set() or {},
        max_async_children=1,
    )
    assert rejected["status"] == "rejected"
    assert "shutting down" in rejected["error"].lower()
    assert started.is_set() is False


def test_interrupt_all_signals_running_children():
    ev = threading.Event()
    interrupted = {"count": 0}
    # No short internal timeout: the blocker holds until interrupt_fn fires.
    # The old ev.wait(timeout=5) made this test a change-detector for CI
    # worker load — on a CPU-starved runner the 5s expired before
    # interrupt_all() ran, the record finalized, and interrupt_all() found
    # nothing running (n == 0). The pytest-level timeout is the real
    # runaway guard.

    def blocker():
        ev.wait(timeout=60)
        return {"status": "interrupted", "summary": None,
                "error": "cancelled"}

    def interrupt_fn():
        interrupted["count"] += 1
        ev.set()

    r = ad.dispatch_async_delegation(
        goal="long task", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=blocker,
        interrupt_fn=interrupt_fn, max_async_children=3,
    )
    n = ad.interrupt_all(reason="test")
    assert n == 1
    assert interrupted["count"] == 1
    # child still emits a completion event after interrupt. Match on THIS
    # delegation's id — straggler 'completed' events from a previous test's
    # workers can finalize after that test's teardown drain and leak into
    # this queue (observed on loaded CI workers).
    evt = _drain_for(r["delegation_id"])
    assert evt is not None
    assert evt["status"] == "interrupted"


def _fast_stale_monitor(monkeypatch, *, idle=0.15, in_tool=0.3, grace=0.15):
    """Shrink the stale-monitor cadence so tests run in milliseconds."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", idle)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", in_tool)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", grace)


def test_stalled_runner_is_interrupted_then_finalized(monkeypatch):
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    interrupted = {"count": 0}

    def stuck_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "too late"}

    def interrupt_fn():
        interrupted["count"] += 1

    res = ad.dispatch_async_delegation(
        goal="stuck child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=stuck_runner,
        interrupt_fn=interrupt_fn, max_async_children=1,
        # Frozen progress token: the child never advances an API call.
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["type"] == "async_delegation"
        assert evt["status"] == "stalled"
        assert evt["delegation_id"] == res["delegation_id"]
        assert evt["api_calls"] == 0
        assert "stalled" in evt["error"]
        # Interrupt was requested BEFORE force-finalization (grace window).
        assert interrupted["count"] >= 1
        assert ad.active_count() == 0
    finally:
        gate.set()

    # If the ignored runner eventually returns, it must not enqueue a second
    # completion for a delegation the monitor already finalized.
    assert _drain_one(timeout=0.5) is None


def test_progressing_runner_is_never_stalled(monkeypatch):
    """A child that keeps advancing is left alone no matter how long it runs."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    ticks = {"n": 0}

    def slow_but_alive_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "done", "api_calls": 7}

    def progress_fn():
        # Token advances on every sample — simulates a child making steady
        # API-call progress.
        ticks["n"] += 1
        return (ticks["n"], None), False

    res = ad.dispatch_async_delegation(
        goal="slow child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=slow_but_alive_runner,
        max_async_children=1, progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    # Run well past the (shrunk) idle threshold — several monitor sweeps.
    time.sleep(0.6)
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"
    assert evt["summary"] == "done"


def test_stalling_runner_that_honors_interrupt_keeps_its_result(monkeypatch):
    """Interrupt-responsive children finalize through the NORMAL path.

    The monitor's interrupt gives a wedged-looking child a grace window; if
    the runner returns during it, the real result (partial work, api_calls)
    is delivered instead of a synthetic stalled event.
    """
    _fast_stale_monitor(monkeypatch, grace=5.0)
    interrupted = threading.Event()

    def runner():
        # "Wedged" until interrupted, then unwinds and reports partial work.
        interrupted.wait(timeout=10)
        return {
            "status": "interrupted",
            "summary": "partial work saved",
            "api_calls": 3,
        }

    res = ad.dispatch_async_delegation(
        goal="responsive child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner,
        interrupt_fn=interrupted.set, max_async_children=1,
        progress_fn=lambda: ((3, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "interrupted"
    assert evt["summary"] == "partial work saved"
    assert evt["api_calls"] == 3
    assert ad.active_count() == 0


def test_streaming_child_counts_as_alive(monkeypatch):
    """A child mid-stream (api_call_count frozen, last_activity_ts ticking)
    must never be stalled — streamed chunks tick _touch_activity, and the
    progress token includes that timestamp (same liveness signal as the
    compaction inactivity budget, PR #71508)."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    now = {"ts": 1000.0}

    def progress_fn():
        # api_call_count and current_tool frozen (long streaming response in
        # flight), but the activity timestamp advances with every chunk.
        now["ts"] += 1.0
        return ((1, None, now["ts"]),), False

    res = ad.dispatch_async_delegation(
        goal="streaming child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: (gate.wait(timeout=10), {"status": "completed", "summary": "streamed"})[1],
        progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    time.sleep(0.6)  # several sweeps past the shrunk idle threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_stalled_event_carries_structured_stall_metadata(monkeypatch):
    """The terminal stalled event must expose machine-readable stall context
    (#51690) — quiet duration, tripped threshold, phase, grace — mirroring
    the sync path's timeout_seconds/timed_out_after_seconds/timeout_phase."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()

    res = ad.dispatch_async_delegation(
        goal="stall metadata", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: ((0, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["status"] == "stalled"
        assert evt["stalled_after_quiet_seconds"] >= 0.3  # in-tool threshold
        assert evt["stall_threshold_seconds"] == ad._STALE_IN_TOOL_SECONDS
        assert evt["stall_phase"] == "in_tool"
        assert evt["stall_grace_seconds"] == ad._STALL_GRACE_SECONDS
    finally:
        gate.set()


def test_list_async_delegations_exposes_live_activity(monkeypatch):
    """list_async_delegations must expose per-child live activity sampled
    from progress_fn plus seconds_since_progress, for /agents UIs (#51690)."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    gate = threading.Event()
    base_ts = time.time() - 12.0

    res = ad.dispatch_async_delegation(
        goal="live listing", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: (((3, "web_search", base_ts),), True),
    )
    try:
        time.sleep(0.1)  # let the monitor stamp _progress_ts at least once
        item = next(
            d for d in ad.list_async_delegations()
            if d["delegation_id"] == res["delegation_id"]
        )
        assert item["status"] == "running"
        assert item["in_tool"] is True
        assert "seconds_since_progress" in item
        (child,) = item["children_activity"]
        assert child["api_calls"] == 3
        assert child["current_tool"] == "web_search"
        assert 10.0 <= child["seconds_since_activity"] <= 20.0
        # Callables and private bookkeeping must never leak.
        assert "progress_fn" not in item
        assert "interrupt_fn" not in item
        assert not any(k.startswith("_") for k in item)
    finally:
        gate.set()


def test_in_tool_stall_uses_higher_threshold(monkeypatch):
    """A frozen child inside a tool gets the in-tool ceiling, not the idle one."""
    _fast_stale_monitor(monkeypatch, idle=0.1, in_tool=10.0, grace=0.1)
    gate = threading.Event()

    def runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "long tool finished"}

    res = ad.dispatch_async_delegation(
        goal="long tool child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner, max_async_children=1,
        # Frozen token but in_tool=True — a legitimately slow terminal
        # command / web fetch. Must NOT be stalled at the idle threshold.
        progress_fn=lambda: ((1, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    time.sleep(0.5)  # far past idle threshold, well under in-tool threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_real_process_restart_restores_owned_completion_once(tmp_path):
    """Real-import E2E: a fresh interpreter restores a prior process's result."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = r'''
import time
from tools import async_delegation as ad
r = ad.dispatch_async_delegation(
    goal="restart", context=None, toolsets=None, role="leaf", model="m",
    session_key="owner-session", parent_session_id="durable-parent",
    runner=lambda: {"status": "completed", "summary": "after restart"},
)
deadline = time.time() + 5
while ad.active_count() and time.time() < deadline:
    time.sleep(.01)
print(r["delegation_id"])
'''
    first = subprocess.run(
        [sys.executable, "-c", producer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    delegation_id = first.stdout.strip().splitlines()[-1]

    consumer = r'''
import json
from tools.process_registry import process_registry
evt = process_registry.completion_queue.get_nowait()
print(json.dumps(evt, sort_keys=True))
'''
    second = subprocess.run(
        [sys.executable, "-c", consumer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    evt = json.loads(second.stdout.strip().splitlines()[-1])
    assert evt["delegation_id"] == delegation_id
    assert evt["session_key"] == "owner-session"
    assert evt["parent_session_id"] == "durable-parent"
    assert evt["summary"] == "after restart"

    acker = f'''
from tools import async_delegation as ad
assert ad.mark_completion_delivered({delegation_id!r})
'''
    subprocess.run(
        [sys.executable, "-c", acker], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "from tools.process_registry import process_registry; print(process_registry.completion_queue.qsize())"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=15, check=True,
    )
    assert probe.stdout.strip().splitlines()[-1] == "0"


# ---------------------------------------------------------------------------
# Integration: delegate_task(background=True) routing
# ---------------------------------------------------------------------------

def test_delegate_task_background_routes_async_and_does_not_block(monkeypatch):
    """delegate_task(background=True) returns a handle without running the
    child synchronously, and the child completes on the background thread.
    A single task is dispatched as a one-item background batch unit."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)  # a sync impl would hang delegate_task here
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    # monkeypatch (not `with`) so patches outlive delegate_task's return and
    # remain active while the background worker runs.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        goal="the real task", context="ctx",
        background=True, parent_agent=parent,
    )

    import json
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: delegate_task returned while the child is STILL
    # blocked on the closed gate, so no completion event exists yet.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1  # one background batch unit, not finished

    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Single task rides the batch path → carries a 1-item results list.
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 1
    assert evt["results"][0]["summary"] == "done: the real task"
    text = format_process_notification(evt)
    assert text is not None
    assert "the real task" in text


def test_delegate_task_background_uses_live_tui_agent_session_id(monkeypatch):
    """TUI async delegation must route to the live/compressed agent id.

    Regression: delegate_task captured the stale approval/session context key
    after compression rotated parent_agent.session_id. The resulting completion
    was orphaned and could be consumed by an unrelated desktop session poller.
    """
    import json
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.approval import reset_current_session_key, set_current_session_key

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "post-compress-tip"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *a, **k: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        },
    )

    approval_token = set_current_session_key("pre-compress-parent")
    session_tokens = set_session_vars(
        source="tui",
        session_key="pre-compress-parent",
        ui_session_id="origin-tab",
    )
    try:
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)
        assert json.loads(out)["status"] == "dispatched"
        evt = _drain_one()
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["session_key"] == "post-compress-tip"
    assert evt["origin_ui_session_id"] == "origin-tab"


def test_concurrent_dispatch_respects_capacity():
    """Two racing dispatches with cap=1 produce one runner and one queued job."""
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    results = []
    barrier = threading.Barrier(2)

    def racer():
        barrier.wait(timeout=5)
        results.append(
            ad.dispatch_async_delegation(
                goal="race", context=None, toolsets=None, role="leaf",
                model="m", session_key="", runner=blocker,
                max_async_children=1,
            )
        )

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["dispatched", "queued"]
    gate.set()
    expected_ids = {result["delegation_id"] for result in results}
    seen_ids = set()
    deadline = time.monotonic() + 5
    while seen_ids != expected_ids and time.monotonic() < deadline:
        event = _drain_one(timeout=0.2)
        if event and event.get("delegation_id") in expected_ids:
            seen_ids.add(event["delegation_id"])
    assert seen_ids == expected_ids


# ---------------------------------------------------------------------------
# Gateway routing: session_key -> platform/chat_id, rich formatting, injection
# ---------------------------------------------------------------------------

def _make_async_evt(**over):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x1",
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "context": "repo /tmp/p",
        "toolsets": ["terminal"],
        "role": "leaf",
        "model": "m",
        "status": "completed",
        "summary": "Found the bug in test_foo",
        "api_calls": 4,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_gateway_formatter_renders_async_block():
    from gateway.run import _format_gateway_process_notification

    txt = _format_gateway_process_notification(_make_async_evt())
    assert txt is not None
    assert "ASYNC DELEGATION COMPLETE" in txt
    assert "Found the bug in test_foo" in txt
    assert "Investigate flaky test" in txt


def test_gateway_cli_origin_event_left_unrouted():
    """An empty session_key (CLI origin) is left without routing fields."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt(session_key="")
    runner._enrich_async_delegation_routing(evt)
    assert "platform" not in evt


def test_single_task_truncation_banner_when_max_iterations():
    """A single async subagent that hit its iteration cap (exit_reason=
    max_iterations) must surface a TRUNCATED marker in the formatted result,
    even though status stays 'completed' (a summary exists)."""
    evt = _make_async_evt(
        status="completed",
        summary="Did part of the work then ran out of budget.",
        exit_reason="max_iterations",
    )
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" in text
    assert "max_iterations" in text
    # The summary is still shown, just flagged.
    assert "Did part of the work" in text


def test_single_task_no_banner_when_clean():
    """A cleanly-finished subagent must NOT get a truncation banner."""
    evt = _make_async_evt(status="completed", summary="All done.", exit_reason="completed")
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" not in text


def test_batch_truncation_banner_marks_only_truncated_task():
    """In a batch, only the task that hit max_iterations gets the TRUNCATED
    marker; a clean sibling keeps the normal check icon."""
    evt = _make_async_evt(
        is_batch=True,
        goals=["clean task", "truncated task"],
        results=[
            {
                "task_index": 0,
                "status": "completed",
                "summary": "finished cleanly",
                "api_calls": 5,
                "exit_reason": "completed",
                "truncated": False,
            },
            {
                "task_index": 1,
                "status": "completed",
                "summary": "cut off mid-work",
                "api_calls": 250,
                "exit_reason": "max_iterations",
                "truncated": True,
            },
        ],
    )
    text = format_process_notification(evt)
    assert text is not None
    assert "TRUNCATED" in text
    # The clean task's summary and the truncated one's both render...
    assert "finished cleanly" in text
    assert "cut off mid-work" in text
    # ...but the banner is tied to the truncated task, not the clean one.
    trunc_pos = text.index("cut off mid-work")
    clean_pos = text.index("finished cleanly")
    banner_pos = text.index("TRUNCATED")
    # The header banner for task 2 appears after task 1's summary.
    assert banner_pos > clean_pos

