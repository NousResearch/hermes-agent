"""Real registry/SQLite coverage for graph capacity, identity and delivery."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("background state did not converge")


@pytest.fixture(autouse=True)
def clean_registry():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    _wait_until(lambda: ad.active_count() == 0)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _batch(index, gate, started, **kwargs):
    def runner():
        started.set()
        assert gate.wait(10)
        return {"results": [{"task_index": index, "status": "completed", "summary": str(index)}]}

    return {
        "goals": [f"Task {index}"], "runner": runner,
        "session_key": "owner", "origin_ui_session_id": "owner-ui",
        "batch_metadata": {"task_indices": [index]},
        **kwargs,
    }


def _single(goal, runner, **kwargs):
    return ad.dispatch_async_delegation(
        goal=goal, context=None, toolsets=None, role="leaf", model="m",
        session_key="other", runner=runner, **kwargs,
    )


def test_graph_occupies_one_slot_without_starving_other_dispatches():
    gate = threading.Event()
    started = [threading.Event() for _ in range(5)]
    try:
        result = ad.dispatch_async_delegation_batches(
            batches=[_batch(i, gate, started[i]) for i in range(3)],
            graph_id="deleg_graph", max_async_children=3,
        )
        assert result["delegation_id"] == "deleg_graph"
        for event in started[:3]:
            assert event.wait(5)
        assert ad.active_count() == 1
        assert ad.active_for_session("owner-ui") == 1
        assert ad.active_task_count() == 3
        assert [row["delegation_id"] for row in ad.list_async_delegations()] == ["deleg_graph"]

        for i in (3, 4):
            accepted = _single(str(i), _batch(i, gate, started[i])["runner"], max_async_children=3)
            assert accepted["status"] == "dispatched"
            # Proves the graph did not consume every physical executor worker.
            assert started[i].wait(5)
        assert ad.active_count() == 3
        rejected = _single("overflow", lambda: pytest.fail("must not run"), max_async_children=3)
        assert rejected["status"] == "rejected"
    finally:
        gate.set()
    events = [process_registry.completion_queue.get(timeout=5) for _ in range(5)]
    assert len({event["delegation_id"] for event in events}) == 5
    assert sum(event.get("graph_id") == "deleg_graph" for event in events) == 3


def test_graph_handle_survives_partial_completion_and_cancels_remaining_clusters(monkeypatch):
    gates = [threading.Event() for _ in range(3)]
    started = [threading.Event() for _ in gates]
    interrupts = [MagicMock(side_effect=gate.set) for gate in gates]
    # Completed components must not be pruned while their graph is still live.
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 0)
    try:
        result = ad.dispatch_async_delegation_batches(
            batches=[_batch(i, gates[i], started[i], interrupt_fn=interrupts[i]) for i in range(3)],
            graph_id="deleg_control", max_async_children=1,
        )
        assert all(event.wait(5) for event in started)
        gates[0].set()
        first = process_registry.completion_queue.get(timeout=5)
        assert first["results"][0]["task_index"] == 0
        _wait_until(lambda: ad.list_async_delegations()[0]["completed_clusters"] == 1)
        snapshot = ad.list_async_delegations()[0]
        assert snapshot["delegation_id"] == result["delegation_id"]
        assert snapshot["status"] == "running"
        assert snapshot["cluster_count"] == 3
        assert ad.active_count() == 1
        assert ad.interrupt_delegation(result["delegation_id"])
        interrupts[0].assert_not_called()
        interrupts[1].assert_called_once()
        interrupts[2].assert_called_once()
    finally:
        for gate in gates:
            gate.set()
    remaining = [process_registry.completion_queue.get(timeout=5) for _ in range(2)]
    assert {event["results"][0]["task_index"] for event in remaining} == {1, 2}


def test_session_interrupt_stops_whole_graph_without_touching_another_session():
    gates = [threading.Event(), threading.Event()]
    other_gate = threading.Event()
    stopped = [MagicMock(side_effect=gate.set) for gate in gates]
    other_stopped = MagicMock(side_effect=other_gate.set)
    try:
        graph = ad.dispatch_async_delegation_batches(
            batches=[_batch(i, gates[i], threading.Event(), interrupt_fn=stopped[i]) for i in range(2)],
            graph_id="deleg_owned", max_async_children=2,
        )
        other = _single(
            "unrelated", _batch(2, other_gate, threading.Event())["runner"],
            interrupt_fn=other_stopped, max_async_children=2,
        )
        assert graph["status"] == other["status"] == "dispatched"
        assert ad.interrupt_for_session(origin_ui_session_id="foreign") == 0
        assert ad.interrupt_for_session(origin_ui_session_id="owner-ui") == 1
        for callback in stopped:
            callback.assert_called_once()
        other_stopped.assert_not_called()
    finally:
        for gate in gates:
            gate.set()
        other_gate.set()


@pytest.mark.parametrize("failure", ["persist", "executor", "submit"])
def test_failed_group_admission_rolls_back_every_component(failure, monkeypatch):
    started = threading.Event()
    if failure == "persist":
        persist = ad._persist_dispatch

        def failing_persist(record):
            persist(record)
            if record["delegation_id"].endswith("_2"):
                raise OSError("test persistence failure")

        monkeypatch.setattr(ad, "_persist_dispatch", failing_persist)
    elif failure == "executor":
        monkeypatch.setattr(ad, "_get_executor", MagicMock(side_effect=RuntimeError("pool creation failed")))
    else:
        executor = MagicMock()
        executor.submit.side_effect = RuntimeError("test submission failure")
        monkeypatch.setattr(ad, "_get_executor", lambda _max: executor)
    result = ad.dispatch_async_delegation_batches(
        batches=[{"goals": [str(i)], "runner": lambda: started.set() or {}} for i in range(2)],
        graph_id="deleg_atomic", max_async_children=1,
    )
    assert result["status"] == "rejected"
    assert not started.is_set()
    assert ad.active_count() == 0
    assert ad.list_async_delegations() == []
    for i in (1, 2):
        assert ad.get_durable_delegation(f"deleg_atomic_cluster_{i}") is None


def test_graph_cannot_combine_different_owners():
    result = ad.dispatch_async_delegation_batches(batches=[
        {"goals": ["a"], "session_key": "a", "runner": lambda: {}},
        {"goals": ["b"], "session_key": "b", "runner": lambda: {}},
    ])
    assert result["status"] == "rejected"
    assert ad.active_count() == 0


def test_restart_replays_components_with_shared_graph_identity_and_separate_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    repo = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = '''
import json, time
from tools import async_delegation as ad
result = ad.dispatch_async_delegation_batches(
    graph_id="deleg_restart", max_async_children=1,
    batches=[{"goals": [str(i)], "session_key": "owner", "runner": lambda: {"results": []}} for i in range(2)],
)
deadline = time.monotonic() + 5
while ad.active_count() and time.monotonic() < deadline:
    time.sleep(.01)
assert ad.active_count() == 0
print(json.dumps(result))
'''
    first = subprocess.run(
        [sys.executable, "-c", producer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    dispatched = json.loads(first.stdout.strip().splitlines()[-1])
    assert dispatched["delegation_id"] == "deleg_restart"
    # This interpreter imports the actual ledger and restores the other
    # process's completions, not fixture-created event dictionaries.
    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 2
    events = [restored.get_nowait() for _ in range(2)]
    assert {event["graph_id"] for event in events} == {"deleg_restart"}
    assert len({event["delegation_id"] for event in events}) == 2
    claims = [ad.claim_event_delivery(event, "test") for event in events]
    assert all(claims)
    for event, claim in zip(events, claims):
        assert ad.claim_event_delivery(event, "another-consumer") is None
        ad.complete_event_delivery(event, claim)
    assert ad.restore_undelivered_completions(queue.Queue()) == 0
