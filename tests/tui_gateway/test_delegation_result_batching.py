"""Subagent results that are ready together are delivered in ONE agent turn.

A fan-out whose units finish while the session is busy piles its results up in the completion
queue. Each used to claim the idle session for a turn of its own, so 45 units meant 45
back-to-back turns, each replaying the whole conversation for a one-line acknowledgement.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


def _delegation(n: int, **extra) -> dict:
    return {"type": "async_delegation", "delegation_id": f"deleg_abc-{n}", "session_key": "stored",
            "goals": [f"audit area {n}"], "total_duration_seconds": 10 * n,
            "results": [{"task_index": 0, "status": "completed", "summary": f"finding {n}"}], **extra}


@pytest.fixture
def harness(monkeypatch):
    submits: list = []
    ledger = {"completed": [], "released": []}
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda rid, sid, session, text, **kw: submits.append((text, kw)))
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery",
                        lambda evt, consumer: f"claim:{evt['delegation_id']}")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery",
                        lambda evt, claim: ledger["completed"].append(claim))
    monkeypatch.setattr("tools.async_delegation.release_event_delivery",
                        lambda evt, claim: ledger["released"].append(claim))
    registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}

    def deliver(events, deferred=None):
        server._notif_handle_ready("sid", session, events, set(), registry,
                                   lambda evt: f"[RESULT {evt['delegation_id']}]", deferred, owned=True)

    return SimpleNamespace(submits=submits, ledger=ledger, registry=registry, session=session, deliver=deliver)


def test_ready_subagent_results_run_as_one_turn(harness):
    harness.deliver([_delegation(1), _delegation(2), _delegation(3)])

    assert len(harness.submits) == 1
    text, kwargs = harness.submits[0]
    assert text == "[RESULT deleg_abc-1]\n\n[RESULT deleg_abc-2]\n\n[RESULT deleg_abc-3]"
    assert kwargs["display_kind"] == "async_delegation_complete"
    meta = kwargs["display_metadata"]
    assert meta["display_text"].startswith("3 Subagent Results: Subagent Task Completed: audit area 1; ")
    assert (meta["task_count"], meta["completed_count"], meta["failed_count"]) == (3, 3, 0)
    assert meta["delegation_id"] == "deleg_abc-1, deleg_abc-2, deleg_abc-3"
    assert meta["duration_seconds"] == 30
    assert harness.ledger == {"completed": ["claim:deleg_abc-1", "claim:deleg_abc-2", "claim:deleg_abc-3"],
                              "released": []}
    assert harness.registry.completion_queue.empty()


def test_a_single_result_keeps_its_own_turn_and_row(harness):
    event = _delegation(7)

    harness.deliver([event])

    assert harness.submits == [("[RESULT deleg_abc-7]", {
        "display_kind": "async_delegation_complete",
        "display_metadata": server._async_delegation_display_metadata(event)})]
    assert harness.ledger["completed"] == ["claim:deleg_abc-7"]


@pytest.mark.parametrize("path", ["poller", "post-turn drain"])
def test_results_ready_while_busy_all_wait_for_the_next_idle_turn(harness, monkeypatch, path):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    harness.session["running"] = True
    events = [_delegation(1), _delegation(2)]
    deferred = [] if path == "post-turn drain" else None

    harness.deliver(events, deferred)

    assert harness.submits == []
    waiting = deferred if deferred is not None else [harness.registry.completion_queue.get_nowait()
                                                     for _ in range(harness.registry.completion_queue.qsize())]
    assert [e["delegation_id"] for e in waiting] == ["deleg_abc-1", "deleg_abc-2"]


def test_a_failed_batch_submit_releases_every_claim_and_the_turn(harness, monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no free worker")))

    harness.deliver([_delegation(1), _delegation(2)])

    assert harness.ledger == {"completed": [], "released": ["claim:deleg_abc-1", "claim:deleg_abc-2"]}
    assert harness.session["running"] is False


def test_early_failure_notices_are_not_merged_into_the_results_turn(harness, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    notice = _delegation(9, task_failure_notice=True)

    harness.deliver([_delegation(1), _delegation(2), notice])

    assert [text for text, _kw in harness.submits] == ["[RESULT deleg_abc-1]\n\n[RESULT deleg_abc-2]"]
    # The session is busy with that turn, so the diagnostic notice waits for the next idle boundary.
    assert harness.registry.completion_queue.get_nowait()["delegation_id"] == "deleg_abc-9"
