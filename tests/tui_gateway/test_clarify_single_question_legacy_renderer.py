"""A one-question clarify batch must stay renderable and answerable by a client that only
speaks the historical single-question shape.

Concrete consumer: the Hermes-Relay Android client
(https://github.com/Codename-11/hermes-relay). Its `GatewayEventMapper.interactionRequest`
reads ONLY `question` / `choices` / `multi_select` off a `clarify.request` payload and falls
back to the literal string "The agent needs clarification" when `question` is absent; it
answers with `{request_id, answer}` and never sends a `question_id`. A `questions`-only
payload therefore renders as a bare text box with the question AND the options lost, and the
reply is discarded.

Two contracts, both proven against the real `tui_gateway.server` bridge:

1. WIRE — the emitted `clarify.request` payload carries `question`/`choices` alongside
   `questions`, so a renderer that never learned the batch shape still draws pickable rows.
2. ANSWER — a reply WITHOUT `question_id` must reach the caller as the answer to the sole
   question, not be parsed as a batch-result dict and dropped (which returned a blank
   `user_response` for every question).
"""

import json
import threading

import pytest

from tui_gateway import server


@pytest.fixture
def clarify_bridge(monkeypatch):
    """Capture emitted events and run `_clarify_block` off-thread so a reply can race in."""
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: emitted.append((event, payload)))
    monkeypatch.setattr(server, "_clarify_timeout_seconds", lambda: 10)
    return emitted


def _one_question(qid="q0"):
    return [{"qid": qid, "question": "Do the options show up as tappable rows?",
             "choices": ["Yes (Recommended)", "No"], "multi_select": False}]


def _run_block(questions):
    """Run the blocking bridge in a worker; return (thread, result_holder)."""
    result: dict[str, str] = {}

    def target():
        result["value"] = server._clarify_block("sid-1", None, None, questions=questions)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


def _wait_for_request(emitted, timeout=5.0):
    deadline = threading.Event()
    for _ in range(int(timeout * 100)):
        if emitted:
            return emitted[0]
        deadline.wait(0.01)
    raise AssertionError("clarify.request was never emitted")


def test_single_question_batch_also_carries_the_simple_shape(clarify_bridge):
    """A legacy renderer reads `question`/`choices`; a batch-aware one still reads `questions`."""
    questions = _one_question()
    thread, _ = _run_block(questions)
    try:
        event, payload = _wait_for_request(clarify_bridge)
        assert event == "clarify.request"
        # Batch-aware clients keep their key.
        assert [q["qid"] for q in payload["questions"]] == ["q0"]
        # Legacy clients find the question AND the choices at the top level.
        assert payload["question"] == questions[0]["question"]
        assert payload["choices"] == questions[0]["choices"]
    finally:
        server._respond(1, {"request_id": payload["request_id"]}, "answer")
        thread.join(timeout=5)


def test_multi_question_batch_keeps_only_the_batch_shape(clarify_bridge):
    """The compatibility keys are meaningless for >1 question and must not be invented."""
    questions = _one_question("q0") + [
        {"qid": "q1", "question": "Second?", "choices": None, "multi_select": False}]
    thread, _ = _run_block(questions)
    try:
        _, payload = _wait_for_request(clarify_bridge)
        assert len(payload["questions"]) == 2
        assert "question" not in payload and "choices" not in payload
    finally:
        server._respond(1, {"request_id": payload["request_id"]}, "answer")
        thread.join(timeout=5)


def test_bare_reply_answers_the_sole_question_instead_of_being_dropped(clarify_bridge):
    """No `question_id` (legacy renderer) still lands as the answer to the only question."""
    thread, result = _run_block(_one_question())
    _, payload = _wait_for_request(clarify_bridge)

    # Exactly what a renderer that never learned `question_id` sends.
    server._respond(1, {"request_id": payload["request_id"], "answer": "Yes (Recommended)"}, "answer")
    thread.join(timeout=5)

    assert json.loads(result["value"]) == {"answers": {"q0": "Yes (Recommended)"}}


def test_bare_empty_reply_is_still_a_cancel(clarify_bridge):
    """An empty bare reply keeps its cancel-all meaning — it is not an answer of ""."""
    thread, result = _run_block(_one_question())
    _, payload = _wait_for_request(clarify_bridge)

    server._respond(1, {"request_id": payload["request_id"], "answer": ""}, "answer")
    thread.join(timeout=5)

    assert result["value"] == ""


def test_question_id_reply_still_locks_per_question(clarify_bridge):
    """The batch-aware path is unchanged: a qid-addressed answer resolves through the registry."""
    thread, result = _run_block(_one_question())
    _, payload = _wait_for_request(clarify_bridge)

    server._respond(1, {"request_id": payload["request_id"], "question_id": "q0",
                        "answer": "No"}, "answer")
    thread.join(timeout=5)

    assert json.loads(result["value"]) == {"answers": {"q0": "No"}}
