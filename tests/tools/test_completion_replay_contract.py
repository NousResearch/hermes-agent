"""Delivery-state contract for completion notifications (stale-replay fix).

The user-visible defect: a completion the agent had already delivered came back on a
later, unrelated task report. This file pins the two halves of the contract that make
that impossible:

* a completion the agent already holds (wait/log, or an inline poll after the exit) is
  not re-announced by ANY surface — the CLI drain, the gateway watcher (covered in
  ``tests/gateway/test_background_process_notifications.py``) or the TUI batch renderer;
* the durable async-delegation ledger replays only completions that were never
  acknowledged, and replay is bounded (age cap) and claim-arbitrated, so a delivered row
  can never come back and a retryable one still gets exactly one retry.
"""

import queue
import time

import pytest

from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
from tools import async_delegation as ad
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import ProcessNotificationBatch

SESSION_KEY = "agent:main:telegram:dm:123:"


@pytest.fixture(autouse=True)
def _clean_ledger():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _registry(monkeypatch) -> ProcessRegistry:
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _register_completion(registry, sid, output="done\n", poll=False):
    session = ProcessSession(
        id=sid, command=f"echo {sid}", task_id="t1", started_at=time.time(),
        exited=False, output_buffer="", notify_on_complete=True, session_key=SESSION_KEY,
    )
    registry._running[sid] = session
    session.mark_exited(0)
    session.output_buffer = output
    registry._move_to_finished(session)
    if poll:
        registry.poll(sid)
    return session


def _seed_ledger(delegation_id, *, age_s=0.0, delivered=False):
    """Create one durable completion row exactly the way dispatch+finalize do."""
    now = time.time()
    ad._persist_dispatch({
        "delegation_id": delegation_id,
        "session_key": SESSION_KEY,
        "parent_session_id": "parent-1",
        "dispatched_at": now - age_s,
        "goal": "goal",
    })
    ad._persist_completion(
        {"type": "async_delegation", "delegation_id": delegation_id, "status": "completed",
         "completed_at": now - age_s, "session_key": SESSION_KEY,
         "parent_session_id": "parent-1", "summary": "done"},
        {"status": "completed", "summary": "done"})
    if delivered:
        assert ad.mark_completion_delivered(delegation_id) is True


# ---------------------------------------------------------------------------
# N1 — one delivery for a new completion
# ---------------------------------------------------------------------------

def test_a_new_completion_is_emitted_exactly_once(monkeypatch):
    registry = _registry(monkeypatch)
    _register_completion(registry, "proc_new")

    first = registry.drain_notifications(session_key=SESSION_KEY, owns_event=lambda e: True)
    assert [evt["session_id"] for evt, _text in first] == ["proc_new"]

    # A second pass over the same completed process has nothing left to hand out.
    assert registry.drain_notifications(session_key=SESSION_KEY, owns_event=lambda e: True) == []
    # And the producer never enqueued a duplicate (kill + reader both finish a session).
    assert registry.completion_queue.empty()


# ---------------------------------------------------------------------------
# N2 — a completion the agent already holds is never re-rendered
# ---------------------------------------------------------------------------

def test_batch_renderer_drops_a_poll_observed_completion(monkeypatch):
    registry = _registry(monkeypatch)
    _register_completion(registry, "proc_polled", poll=True)
    _register_completion(registry, "proc_fresh")

    drained = registry.drain_notifications(
        session_key=SESSION_KEY, owns_event=lambda e: True, skip_poll_observed=False)
    batch = ProcessNotificationBatch(tuple(drained))

    assert [evt["session_id"] for evt, _text in batch._live(registry)] == ["proc_fresh"]
    rendered = batch.render(registry)
    assert "proc_fresh" in rendered
    assert "proc_polled" not in rendered


# ---------------------------------------------------------------------------
# N4 / N6 — durable replay: delivered rows never come back, pending ones do
# ---------------------------------------------------------------------------

def test_a_delivered_row_is_never_replayed():
    _seed_ledger("deleg-delivered", delivered=True)
    target = queue.Queue()

    assert ad.restore_undelivered_completions(target) == 0
    assert target.empty()


def test_a_pending_row_is_replayed_once_and_the_ack_stops_the_replay():
    _seed_ledger("deleg-pending")
    first = queue.Queue()
    assert ad.restore_undelivered_completions(first) == 1
    event = first.get_nowait()
    assert event["delegation_id"] == "deleg-pending"
    assert event["restored"] is True, "restored payloads must prove ownership before delivery"

    # The consumer acknowledges admission through the durable claim API…
    claim = ad.claim_event_delivery(event, "consumer-1")
    assert claim
    ad.complete_event_delivery(event, claim)
    assert ad.get_durable_delegation("deleg-pending")["delivery_state"] == "delivered"

    # …so the next process start has nothing to replay.
    second = queue.Queue()
    assert ad.restore_undelivered_completions(second) == 0
    assert second.empty()


def test_an_unacknowledged_row_stays_retryable_at_least_once():
    """The crash window the design accepts: a completion injected but not yet acked is
    replayed once on restart (at-least-once), never dropped."""
    _seed_ledger("deleg-crash")
    target = queue.Queue()
    assert ad.restore_undelivered_completions(target) == 1
    assert target.qsize() == 1
    assert ad.get_durable_delegation("deleg-crash")["delivery_state"] == "pending"


def test_a_stale_backlog_row_is_terminally_dropped_instead_of_replayed():
    _seed_ledger("deleg-ancient", age_s=ad._MAX_COMPLETION_REPLAY_AGE_S + 3600)
    target = queue.Queue()

    assert ad.restore_undelivered_completions(target) == 0
    assert target.empty()
    assert ad.get_durable_delegation("deleg-ancient")["delivery_state"] == "dropped"


# ---------------------------------------------------------------------------
# N5 — one completion, one consumer
# ---------------------------------------------------------------------------

def test_concurrent_consumers_cannot_both_claim_a_completion():
    _seed_ledger("deleg-race")
    assert ad.claim_completion_delivery("deleg-race", "consumer-a") is True
    assert ad.claim_completion_delivery("deleg-race", "consumer-b") is False
    # Releasing the loser's claim attempt leaves the winner's claim intact.
    assert ad.release_completion_delivery("deleg-race", "consumer-b") is False
    assert ad.complete_completion_delivery("deleg-race", "consumer-a") is True
    assert ad.get_durable_delegation("deleg-race")["delivery_state"] == "delivered"


# ---------------------------------------------------------------------------
# End-to-end A/B/C — the reported user flow
# ---------------------------------------------------------------------------

class _Surface(CLIProcessNotificationsMixin):
    """Minimal live CLI/TUI session on the production mixin."""

    def __init__(self):
        self.session_id = SESSION_KEY
        self._session_db = None
        self._pending_input = queue.Queue()

    def report_pass(self, consumer):
        self._drain_process_notifications(consumer)
        delivered, texts = [], []
        while not self._pending_input.empty():
            item = self._pending_input.get_nowait()
            if isinstance(item, ProcessNotificationBatch):
                import tools.process_registry as pr_module

                delivered.extend(evt["session_id"] for evt, _t in item._live(pr_module.process_registry))
                rendered = item.render(pr_module.process_registry)
                if rendered:
                    texts.append(rendered)
            else:
                texts.append(item)
        return delivered, texts


def test_end_to_end_abc_run_has_no_stale_replay(monkeypatch):
    """Task §9: RUN1 = [A], RUN2 = [B], RUN3 = []."""
    import tools.process_registry as pr_module

    registry = _registry(monkeypatch)
    surface = _Surface()

    _register_completion(registry, "proc_A", "A: build ok\n")
    run1, _ = surface.report_pass("cli:RUN1")

    _register_completion(registry, "proc_B", "B: tests ok\n")
    run2, _ = surface.report_pass("cli:RUN2")

    run3, texts3 = surface.report_pass("cli:RUN3")

    assert run1 == ["proc_A"]
    assert run2 == ["proc_B"], "a delivered completion must not ride along with the next report"
    assert run3 == []
    assert texts3 == []
    assert registry.completion_queue.empty()
