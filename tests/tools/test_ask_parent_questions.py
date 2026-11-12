"""The pending-question store behind ask_parent: one child question waits for the master's answer,
a timeout, or a stop, and always ends with an explicit outcome (never invented answer text)."""
import threading

import pytest

from tests.waiting import wait_until


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    from tools import ask_parent

    monkeypatch.setattr(ask_parent, "_questions", {})
    monkeypatch.setattr(ask_parent, "_asked", {})


def _ask(subagent_id="sa-1", question="Which branch should I target?", **kwargs):
    from tools.ask_parent import open_question

    return open_question(subagent_id, question=question, state=kwargs.pop("state", "Tests pass on main."),
                         options=kwargs.pop("options", None), max_questions=kwargs.pop("max_questions", 3))


def _wait_in_thread(question, **kwargs):
    from tools.ask_parent import wait_for_answer

    out = {}
    kwargs.setdefault("timeout", 5)
    worker = threading.Thread(target=lambda: out.update(wait_for_answer(question, **kwargs)))
    worker.start()
    return worker, out


def test_an_answer_resumes_the_waiting_child_with_the_text():
    from tools.ask_parent import answer_question

    question, error = _ask()
    assert error is None
    worker, out = _wait_in_thread(question)

    assert answer_question("sa-1", question.request_id, "Target the release branch.") is None
    worker.join(3)

    assert out["outcome"] == "answered"
    assert out["answer"] == "Target the release branch."
    assert out["request_id"] == question.request_id


def test_no_answer_in_time_is_timed_out_without_answer_text():
    from tools.ask_parent import answer_question, wait_for_answer

    question, _ = _ask()

    result = wait_for_answer(question, timeout=0.05)

    assert result["outcome"] == "timed_out"
    assert result["answer"] is None
    assert answer_question("sa-1", question.request_id, "too late") is not None  # refused


def test_a_stopped_child_stops_waiting_as_abandoned():
    stop = threading.Event()
    question, _ = _ask()
    worker, out = _wait_in_thread(question, should_abandon=stop.is_set, poll_seconds=0.01)

    stop.set()
    worker.join(3)

    assert out["outcome"] == "abandoned"
    assert out["answer"] is None


def test_abandoning_a_subagent_ends_its_open_question():
    from tools.ask_parent import abandon_questions

    question, _ = _ask()
    worker, out = _wait_in_thread(question)

    abandon_questions("sa-1")
    worker.join(3)

    assert out["outcome"] == "abandoned"


def test_a_child_has_one_open_question_at_a_time():
    first, _ = _ask()

    second, error = _ask(question="And another thing?")

    assert second is None
    assert first.request_id in error


def test_a_child_may_ask_only_so_many_questions():
    from tools.ask_parent import wait_for_answer

    for _ in range(2):
        question, error = _ask(max_questions=2)
        assert error is None
        wait_for_answer(question, timeout=0.01)  # each one ends (timed out) before the next

    question, error = _ask(max_questions=2)

    assert question is None
    assert "2" in error


def test_an_answer_must_name_the_open_question_of_that_subagent():
    from tools.ask_parent import answer_question

    question, _ = _ask()
    _ask(subagent_id="sa-2")

    assert answer_question("sa-1", "req-unknown", "x") is not None
    assert answer_question("sa-2", question.request_id, "x") is not None  # another child's question
    assert answer_question("sa-1", question.request_id, "   ") is not None  # empty answer
    assert answer_question("sa-1", question.request_id, "ok") is None


def test_a_waiting_child_keeps_reporting_activity():
    ticks = []
    question, _ = _ask()
    worker, out = _wait_in_thread(question, on_tick=lambda: ticks.append(1), tick_seconds=0.01, poll_seconds=0.01)

    wait_until(lambda: len(ticks) >= 3, timeout=3)
    from tools.ask_parent import answer_question
    answer_question("sa-1", question.request_id, "go on")
    worker.join(3)

    assert out["outcome"] == "answered"


def test_the_open_question_is_visible_until_it_ends():
    from tools.ask_parent import answer_question, pending_question

    question, _ = _ask(state="Halfway through the migration.", options=["main", "release"])

    shown = pending_question("sa-1")
    assert shown["request_id"] == question.request_id
    assert shown["question"] == "Which branch should I target?"
    assert shown["options"] == ["main", "release"]

    answer_question("sa-1", question.request_id, "main")

    assert pending_question("sa-1") is None
