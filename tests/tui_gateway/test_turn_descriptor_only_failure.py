"""Descriptor-only terminal failures report an error, not an empty complete turn (#135958).

Advisory loop exits such as ``rebuilt_restart_limit_exceeded`` stamp
``failure_reason``/``failure_retryable`` but never set ``failed``/``error`` (cron and the
kanban breaker key on ``failed``). With the completion explainer off the reply slot stays
empty, and the reducer used to report ``status=complete error_retained=False`` for a turn
that produced no answer — clearing the inflight snapshot a reconnect would have replayed.
"""

import contextlib
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv


def _restart_exhausted_result():
    return {
        "final_response": "",
        "completed": False,
        "failed": False,
        "failure_reason": "loop_error",
        "failure_retryable": True,
        "turn_exit_reason": "rebuilt_restart_limit_exceeded",
    }


# ---- reducer boundary (the issue's deterministic reproduction) --------------------------------

def test_restart_exhausted_empty_turn_is_not_complete():
    assert srv._result_status(_restart_exhausted_result()) == "error"


def test_failed_without_error_string_is_not_complete():
    assert srv._result_status({"final_response": "", "completed": False, "failed": True}) == "error"


def test_successful_answer_stays_complete():
    assert srv._result_status({"final_response": "answer", "completed": True}) == "complete"


def test_interruption_stays_interrupted():
    result = {**_restart_exhausted_result(), "interrupted": True}
    assert srv._result_status(result) == "interrupted"


def test_max_iteration_summary_handoff_stays_complete():
    # ``is_max_iteration_handoff``'s shape: incomplete but resumable, with a visible summary
    # and no failure descriptor — never converted into a provider failure.
    result = {
        "final_response": "Part 1 done; send continue for the rest.",
        "completed": False,
        "failed": False,
        "turn_exit_reason": "max_iterations_reached(40/40)",
    }
    assert srv._result_status(result) == "complete"


def test_advisory_exit_with_visible_reasoning_text_stays_complete():
    # ``empty_response_exhausted`` is advisory because the reasoning-only text may literally
    # be the answer: a non-empty reply keeps its turn out of the failure branch.
    result = {
        "final_response": "the answer in reasoning text",
        "completed": False,
        "failed": False,
        "failure_reason": "empty_response",
    }
    assert srv._result_status(result) == "complete"


def test_turn_outcome_carries_recovery_copy_for_descriptor_only_failure():
    raw, status, _ = srv._turn_outcome(_restart_exhausted_result(), None)
    assert status == "error"
    assert raw and "not answered" in raw  # plain account + next step, not an empty frame


# ---- caller boundary: inflight retention and terminal callback --------------------------------

class _Agent:
    provider = "github-copilot"
    model = "gpt-6.1-sol"
    session_id = "s1"


def _turn(result, terminal_callback=None):
    return SimpleNamespace(
        result=result, agent=_Agent(), terminal_callback=terminal_callback,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping",
    )


def _payload(monkeypatch, result, terminal_callback=None):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    session = {
        "pending_title": None, "session_key": "k",
        "history_lock": contextlib.nullcontext(), "agent": _Agent(),
        "inflight_turn": {"assistant": "", "user": "ping", "started_at": 0.0},
    }
    st = _turn(result, terminal_callback)
    payload, raw, status = srv._complete_turn_payload(session, st, None, 80)
    return payload, st, session


def test_complete_turn_payload_retains_inflight_failure(monkeypatch):
    receipts = []
    payload, st, session = _payload(monkeypatch, _restart_exhausted_result(), lambda r: receipts.append(r))

    assert payload["status"] == "error"
    assert payload["error"] and payload["recoverable"] is True
    # Specific descriptor from the stamped failure_reason, so the client renders recovery
    # without re-parsing text.
    assert payload["error_surface"]["code"] == "loop_error"
    assert st.error_retained is True
    inflight = session["inflight_turn"]
    assert inflight["status"] == "error" and inflight["recoverable"] is True
    assert inflight["error"] != "None"  # no error object: generic verdict, not the string "None"
    assert inflight["error_surface"]["code"] == "loop_error"
    assert len(receipts) == 1
    assert receipts[0]["status"] == "failed"
    assert receipts[0]["error"] == payload["error"]


@pytest.mark.parametrize("text", ["", " ", "\n\t", " \n\t "])
def test_blank_failure_has_visible_recovery_copy(monkeypatch, text):
    result = {**_restart_exhausted_result(), "final_response": text}
    receipts = []
    payload, st, session = _payload(monkeypatch, result, receipts.append)

    assert payload["status"] == "error"
    assert payload["text"].strip()
    assert payload["error"].strip()
    assert st.error_retained is True
    assert session["inflight_turn"]["error_surface"]["code"] == "loop_error"
    assert len(receipts) == 1
    assert receipts[0]["status"] == "failed"
    assert receipts[0]["text"].strip()
    assert receipts[0]["error"].strip()


@pytest.mark.parametrize("partial", [False, True])
def test_explicit_failure_preserves_visible_text_contract(monkeypatch, partial):
    result = {
        "final_response": "Visible failure explanation or partial answer",
        "completed": False,
        "failed": True,
        "partial": partial,
    }
    receipts = []
    payload, st, session = _payload(monkeypatch, result, receipts.append)

    assert payload["status"] == "error"
    assert payload["text"] == result["final_response"]
    assert st.error_retained is True
    assert receipts[0]["status"] == "failed"
    assert bool(payload.get("partial")) is partial
    assert session["inflight_turn"]["assistant"] == (result["final_response"] if partial else "")


def test_complete_turn_payload_successful_turn_still_clears_inflight(monkeypatch):
    payload, st, session = _payload(monkeypatch, {"final_response": "answer", "completed": True})

    assert payload["status"] == "complete"
    assert st.error_retained is False
    assert session["inflight_turn"] is None


def test_fail_inflight_turn_without_error_object_uses_generic_verdict():
    session = {"inflight_turn": None, "history_lock": contextlib.nullcontext()}
    with session["history_lock"]:
        srv._fail_inflight_turn(session, None, error_surface={"layer": "provider", "code": "loop_error"})
    assert session["inflight_turn"]["error"] == "turn failed"
