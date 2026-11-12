"""A subagent's question reaches its live parent as a labelled interim event of its background unit.

It rides the same ``type="async_delegation"`` path as the early task-failure notice, so it wakes an
idle parent as a new turn, but it is never the unit's durable completion: it must not claim,
acknowledge or dedup against the final result, nor against another question.
"""
import queue
from unittest.mock import patch

from tools import async_delegation as ad
from tools.process_registry_notifications import format_process_notification
from tui_gateway.session_notifications import _notification_event_dedup_key


def _record(status="running"):
    return {"delegation_id": "deleg_x", "status": status, "is_batch": True, "goals": ["a", "Port the parser", "c"],
            "goal": "a", "session_key": "sk", "origin_ui_session_id": "ui", "origin_session_id": "", "parent_session_id": "root",
            "dispatched_at": 1.0, "role": "leaf", "model": "m", "context": None, "toolsets": None}


def _question(**overrides):
    return {"subagent_id": "sa-1-abc", "request_id": "req-1", "task_index": 1, "goal": "Port the parser",
            "question": "Keep the old API or break it?", "state": "New parser passes 40/42 tests.",
            "options": ["keep", "break"], "wait_seconds": 1800, **overrides}


def _push(record_status="running", **overrides):
    q = queue.Queue()
    with patch.object(ad, "_records", {"deleg_x": _record(record_status)}), \
         patch("tools.process_registry.process_registry") as reg:
        reg.completion_queue = q
        queued = ad.push_parent_question("deleg_x", _question(**overrides))
    return queued, q


def test_a_question_reaches_the_parent_queue_with_the_unit_routing():
    queued, q = _push()

    assert queued is True
    evt = q.get_nowait()
    assert evt["type"] == "async_delegation" and evt["parent_question"] is True
    assert (evt["session_key"], evt["origin_ui_session_id"], evt["parent_session_id"]) == ("sk", "ui", "root")
    assert (evt["subagent_id"], evt["request_id"]) == ("sa-1-abc", "req-1")


def test_no_question_is_queued_once_the_unit_has_finished():
    queued, q = _push(record_status="completed")

    assert queued is False
    assert q.empty()


def test_no_question_is_queued_when_results_only_reach_the_session_history():
    """An API-server session gets detached results as history rows for its next client turn: no model
    wakes up to answer, so the child must learn at once that its parent is unavailable."""
    q = queue.Queue()
    record = {**_record(), "origin_session_id": "api-session"}
    with patch.object(ad, "_records", {"deleg_x": record}), patch("tools.process_registry.process_registry") as reg:
        reg.completion_queue = q
        queued = ad.push_parent_question("deleg_x", _question())

    assert queued is False
    assert q.empty()


def test_the_parent_reads_a_labelled_question_with_how_to_reply():
    _, q = _push()

    text = format_process_notification(q.get_nowait())

    assert text.startswith("[SUBAGENT QUESTION — from subagent sa-1-abc, task 2/3 of deleg_x; not from the user]")
    assert "Task: Port the parser" in text
    assert "Question: Keep the old API or break it?" in text
    assert "New parser passes 40/42 tests." in text
    assert "keep" in text and "break" in text
    assert 'delegate_task(action="reply", subagent_id="sa-1-abc", request_id="req-1"' in text
    assert "30 min" in text
    assert "not the user's approval" in text


def test_a_question_is_an_interim_event_that_never_claims_the_final_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    _, q = _push()
    question = q.get_nowait()

    assert ad.is_interim_delegation_event(question)
    assert ad.claim_event_delivery(question, "tui-poller") == ""


def test_questions_keep_their_own_identity_at_every_consumer():
    from gateway.run_notifications import GatewayNotificationsMixin

    _, q = _push()
    first = q.get_nowait()
    second = {**first, "request_id": "req-2"}
    final = {"type": "async_delegation", "delegation_id": "deleg_x", "is_batch": True, "results": []}
    failure = {"type": "async_delegation", "delegation_id": "deleg_x", "task_failure_notice": True,
               "results": [{"task_index": 1}]}

    gateway_identity = GatewayNotificationsMixin._completion_delivery_identity
    assert len({gateway_identity(e) for e in (first, second, final, failure)}) == 4
    assert len({_notification_event_dedup_key(e) for e in (first, second, final, failure)}) == 4


def test_the_gateway_never_claims_the_unit_durable_row_for_a_question(monkeypatch):
    """Claiming the row would acknowledge the unit's FINAL result before it exists."""
    import asyncio
    from types import SimpleNamespace

    from gateway.run_notifications import GatewayNotificationsMixin

    claimed = []
    monkeypatch.setattr(ad, "claim_completion_delivery", lambda delegation_id, claim_id: claimed.append(delegation_id) or True)

    async def ready(evt):
        return True

    gateway = SimpleNamespace(_CompletionClaim=GatewayNotificationsMixin._CompletionClaim, _completion_delivery_ready=ready)
    _, q = _push()
    question = {**q.get_nowait(), "parent_session_id": ""}

    claim = asyncio.run(GatewayNotificationsMixin._preflight_completion_delivery(gateway, question))

    assert claimed == []
    assert claim.delegation_id == "" and claim.proceed


def test_the_timeline_shows_the_question_not_a_task_outcome():
    from tools.process_registry_notifications import async_delegation_display_text

    _, q = _push()

    assert async_delegation_display_text(q.get_nowait()) == "Subagent Question: Keep the old API or break it?"


def test_a_question_is_not_presented_as_a_diagnostic():
    from agent.notification_presentation import diagnostic_process_event

    _, q = _push()

    assert not diagnostic_process_event(q.get_nowait())
