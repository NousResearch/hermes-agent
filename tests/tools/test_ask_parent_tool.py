"""The child side of ask_parent: a delegated child pauses on its question until the parent answers,
the wait times out, or it is stopped; with no live parent it learns so at once."""
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until


@pytest.fixture
def setup(monkeypatch):
    from tools import ask_parent, async_delegation, delegate_tool_config
    from tools import delegate_tool_registry as registry

    settings = {"enabled": True, "timeout_seconds": 5}
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"ask_parent": settings})
    monkeypatch.setattr(ask_parent, "_questions", {})
    monkeypatch.setattr(ask_parent, "_asked", {})
    monkeypatch.setattr(ask_parent, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(registry, "_active_subagents", {})
    events = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry.completion_queue", events)
    monkeypatch.setattr(async_delegation, "_records", {"deleg_x": {
        "delegation_id": "deleg_x", "status": "running", "goals": ["Port the parser"], "goal": "Port the parser",
        "session_key": "", "origin_ui_session_id": "", "origin_session_id": "", "parent_session_id": "root"}})
    touched = []
    child = SimpleNamespace(_delegate_depth=1, _subagent_id="sa-1", _async_unit_id="deleg_x", _interrupt_requested=False,
                            _delegate_task_index=0, _touch_activity=lambda desc, **kw: touched.append(desc),
                            tools=[], valid_tool_names=set())
    registry._register_subagent({"subagent_id": "sa-1", "goal": "Port the parser", "status": "running", "agent": child})
    return SimpleNamespace(settings=settings, events=events, child=child, touched=touched, registry=registry)


def _ask(child, **args):
    from tools.ask_parent import ask_parent_tool

    return json.loads(ask_parent_tool(args.pop("question", "Keep the old API?"), agent=child, **args))


def _ask_in_thread(child, **args):
    out = {}
    worker = threading.Thread(target=lambda: out.update(_ask(child, **args)), daemon=True)
    worker.start()
    return worker, out


def test_the_child_pauses_until_the_parent_answers_then_carries_on_with_the_answer(setup):
    from tools.ask_parent import answer_question

    worker, out = _ask_in_thread(setup.child, state="40/42 tests pass.", options=["keep", "break"])
    question = setup.events.get(timeout=3)
    assert question["parent_question"] and question["question"] == "Keep the old API?"
    assert question["state"] == "40/42 tests pass." and question["options"] == ["keep", "break"]
    assert worker.is_alive()  # paused

    answer_question("sa-1", question["request_id"], "Keep it; add a deprecation warning.")
    worker.join(3)

    assert out["outcome"] == "answered"
    assert out["answer"] == "Keep it; add a deprecation warning."
    assert "not" in out["note"] and "user" in out["note"]  # guidance, not the user's approval


def test_the_subagent_shows_as_awaiting_its_parent_while_paused(setup):
    from tools.ask_parent import answer_question

    worker, _ = _ask_in_thread(setup.child)
    question = setup.events.get(timeout=3)
    record = setup.registry._active_subagents["sa-1"]

    assert record["status"] == "awaiting_parent"
    answer_question("sa-1", question["request_id"], "yes")
    worker.join(3)
    assert record["status"] == "running"


def test_without_a_live_parent_the_child_learns_at_once(setup):
    setup.child._async_unit_id = None  # a joined call: the parent is blocked waiting for this result

    out = _ask(setup.child)

    assert out["outcome"] == "parent_unavailable" and out["answer"] is None
    assert setup.events.empty()
    from tools.ask_parent import _asked
    assert _asked.get("sa-1", 0) == 0  # did not use up an allowed question


def test_a_finished_unit_means_no_live_parent(setup):
    from tools import async_delegation

    async_delegation._records["deleg_x"]["status"] = "completed"

    assert _ask(setup.child)["outcome"] == "parent_unavailable"


def test_a_stop_while_paused_ends_the_wait_as_abandoned(setup):
    worker, out = _ask_in_thread(setup.child)
    setup.events.get(timeout=3)

    setup.child._interrupt_requested = True
    worker.join(3)

    assert out["outcome"] == "abandoned" and out["answer"] is None


def test_no_reply_in_time_lets_the_child_carry_on(setup):
    setup.settings["timeout_seconds"] = 0.05

    worker, out = _ask_in_thread(setup.child)
    worker.join(5)

    assert not worker.is_alive(), "the configured wait was not honoured"
    assert out["outcome"] == "timed_out" and out["answer"] is None
    assert "assum" in out["note"]  # carry on, and say what was assumed


def test_a_paused_child_keeps_reporting_activity(setup, monkeypatch):
    from tools import ask_parent

    monkeypatch.setattr(ask_parent, "_TICK_SECONDS", 0.01)
    worker, _ = _ask_in_thread(setup.child)
    question = setup.events.get(timeout=3)

    wait_until(lambda: len(setup.touched) >= 2, timeout=3)
    ask_parent.answer_question("sa-1", question["request_id"], "ok")
    worker.join(3)


def test_only_an_enabled_delegated_child_may_ask(setup):
    setup.settings["enabled"] = False
    assert "error" in _ask(setup.child)

    setup.settings["enabled"] = True
    top_level = SimpleNamespace(_delegate_depth=0, _subagent_id=None)
    assert "error" in _ask(top_level)
    assert setup.events.empty()


def test_an_empty_or_oversized_question_is_refused(setup):
    assert "error" in _ask(setup.child, question="   ")
    assert "error" in _ask(setup.child, question="x" * 5000)
    assert "error" in _ask(setup.child, state="y" * 20000)
    assert setup.events.empty()
