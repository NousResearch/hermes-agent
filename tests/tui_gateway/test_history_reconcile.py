"""Regression test for the Desktop/TUI history-hollow fix.

Root cause (HISTORY_PIPELINE_BUG_CONFIRMED): the Desktop/TUI gateway feeds
``run_conversation`` an in-memory ``session["history"]`` that can be hollowed
across turns (prior-turn content arrives empty while the session DB still holds
the real text). The CLI surface loads history from
``SessionDB.get_messages_as_conversation`` every turn and always carries real
content. ``_reconcile_history_with_session_db`` makes the Desktop path do the
same: when the in-memory copy is hollow, fall back to the DB-backed transcript.

These tests exercise the REAL helper (and the exact history-handling sequence
inside ``_run_prompt_submit``) with a fake DB/agent so they run offline and fast.
"""
import threading
import types

import pytest

from tui_gateway import server as S


class FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def get_messages_as_conversation(self, session_id, include_ancestors=False,
                                      repair_alternation=False, include_row_ids=False):
        return [dict(r) for r in self._rows]


REAL = [
    {"role": "user", "content": "remember the code is 4271"},
    {"role": "assistant", "content": "Got it, code 4271 noted."},
]


def _agent(db):
    a = types.SimpleNamespace()
    a._session_db = db
    a.session_id = "sess_key"
    return a


def _session():
    return {"session_key": "sess_key", "history": [], "history_lock": threading.Lock()}


def test_intact_history_returned_unchanged():
    agent = _agent(FakeDB(REAL))
    sess = _session()
    hist = [dict(r) for r in REAL]
    assert S._reconcile_history_with_session_db(agent, sess, hist) == REAL


def test_hollow_history_filled_from_db():
    agent = _agent(FakeDB(REAL))
    sess = _session()
    hollow = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    out = S._reconcile_history_with_session_db(agent, sess, hollow)
    assert out == REAL
    assert out[0]["content"] == "remember the code is 4271"


def test_hollow_with_unsaved_tail_preserved():
    agent = _agent(FakeDB(REAL))
    sess = _session()
    hollow = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Got it, code 4271 noted."},
        {"role": "user", "content": "what was the code?"},  # unsaved tail
    ]
    out = S._reconcile_history_with_session_db(agent, sess, hollow)
    assert len(out) == 3
    assert out[0]["content"] == "remember the code is 4271"
    assert out[2]["content"] == "what was the code?"


def test_empty_history_unchanged():
    agent = _agent(FakeDB(REAL))
    sess = _session()
    assert S._reconcile_history_with_session_db(agent, sess, []) == []


def test_tool_call_entry_not_treated_hollow():
    agent = _agent(FakeDB(REAL))
    sess = _session()
    hist = [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}]
    assert S._reconcile_history_with_session_db(agent, sess, hist) == hist


def test_db_unavailable_falls_back_to_in_memory():
    agent = _agent(None)  # no _session_db
    sess = _session()
    hollow = [{"role": "user", "content": ""}]
    # No DB -> return the (hollow) in-memory history unchanged rather than crash.
    assert S._reconcile_history_with_session_db(agent, sess, hollow) == hollow


class StubAgent:
    def __init__(self, db):
        self._session_db = db
        self.session_id = "sess_key"
        self.recorded = []

    def run_conversation(self, run_message, **kwargs):
        ch = kwargs.get("conversation_history") or []
        self.recorded.append([dict(m) for m in ch])
        user_msg = {"role": "user", "content": run_message}
        asst_msg = {"role": "assistant", "content": f"echo: {run_message}"}
        return {
            "final_response": asst_msg["content"],
            "messages": [dict(m) for m in ch] + [user_msg, asst_msg],
        }


def _run_turn(agent, session, text):
    # Exact history sequence from _run_prompt_submit.
    with session["history_lock"]:
        history = list(session["history"])
    history = S._reconcile_history_with_session_db(agent, session, history)
    result = agent.run_conversation(text, conversation_history=list(history))
    with session["history_lock"]:
        session["history"] = result["messages"]
    return result


def test_two_turn_hollow_history_reaches_provider():
    agent = StubAgent(FakeDB(REAL))
    session = _session()
    _run_turn(agent, session, "hello")
    assert agent.recorded[0] == []
    # Simulate the bug: gateway in-memory history hollowed.
    session["history"] = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    _run_turn(agent, session, "what was the code?")
    got = agent.recorded[1]
    assert got == REAL
    assert session["history"][0]["content"] == "remember the code is 4271"
