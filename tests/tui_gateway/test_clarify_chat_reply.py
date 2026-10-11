"""A typed chat reply must be able to answer the pending ``clarify`` server request (#134230).

The messaging-gateway route intercepts typed text before busy routing (``run_inbound._hm_clarify_reply``);
the TUI/desktop bridge had no equivalent, so a reply typed into the chat input steered or queued behind
the turn while the clarify card waited out its full timeout and the tool returned an empty
``user_response``. These tests pin the routing helper: selection-shaped replies resolve the request
(open-ended accepts any text), out-of-range selections keep the card armed, and prose on a single
choice prompt releases the waiting tool empty so the message falls through to the normal busy path.
"""

import json
import threading
import types

import pytest

from tui_gateway import server, server_requests


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def _open_clarify(sid: str, params: dict, qids=None):
    req = server_requests.ServerRequest(sid, "clarify", params, qids=qids)
    with server_requests._lock:
        server_requests._open[req.id] = req
    return req


@pytest.fixture(autouse=True)
def _clean_registry():
    server_requests.reset_for_tests()
    server._sessions.pop("clarify-chat-sid", None)
    yield
    server_requests.reset_for_tests()
    server._sessions.pop("clarify-chat-sid", None)


def _answer(text, session=None, sid="clarify-chat-sid", **kw):
    return server._answer_pending_clarify_from_chat(
        "rid", sid, session or _session(), text, **kw
    )


def test_numeric_pick_resolves_single_choice_prompt():
    req = _open_clarify(
        "clarify-chat-sid", {"question": "Which?", "choices": ["a", "b"]}
    )
    resp = _answer("1")
    assert resp and resp["result"]["status"] == "clarify_answered"
    assert req.event.is_set() and req.answered
    assert req.result == {"answer": "a"}


def test_label_pick_resolves_single_choice_prompt():
    req = _open_clarify(
        "clarify-chat-sid", {"question": "Which?", "choices": ["staging", "prod"]}
    )
    resp = _answer("prod")
    assert resp and resp["result"]["status"] == "clarify_answered"
    assert req.result == {"answer": "prod"}


def test_any_text_resolves_open_ended_prompt():
    req = _open_clarify("clarify-chat-sid", {"question": "Name?", "choices": None})
    resp = _answer("Blue database, please")
    assert resp and resp["result"]["status"] == "clarify_answered"
    assert req.result == {"answer": "Blue database, please"}


def test_multi_select_list_returns_json_array():
    req = _open_clarify(
        "clarify-chat-sid",
        {"question": "Which?", "choices": ["a", "b"], "multi_select": True},
    )
    resp = _answer("1,2")
    assert resp and resp["result"]["status"] == "clarify_answered"
    assert json.loads(req.result["answer"]) == ["a", "b"]


def test_out_of_range_pick_keeps_card_armed():
    req = _open_clarify(
        "clarify-chat-sid", {"question": "Which?", "choices": ["a", "b"]}
    )
    resp = _answer("7")
    assert resp and resp["result"]["status"] == "clarify_retry"
    assert not req.event.is_set(), (
        "an out-of-range selection must not settle the clarify"
    )


def test_prose_on_single_choice_prompt_releases_tool_and_falls_through():
    """Free prose on a choice prompt resolves the clarify empty (deadlock break) and returns None
    so the message takes the normal busy path — the gateway's TEXT_REJECTED_PROSE contract."""
    req = _open_clarify(
        "clarify-chat-sid", {"question": "Which?", "choices": ["a", "b"]}
    )
    assert _answer("let's use staging instead") is None
    assert req.event.is_set() and req.answered
    assert req.result == {"answer": ""}


def test_batch_reply_locks_first_unanswered_question_then_settles():
    req = _open_clarify(
        "clarify-chat-sid",
        {
            "questions": [
                {
                    "qid": "q0",
                    "question": "Env?",
                    "choices": ["staging", "prod"],
                    "multi_select": False,
                },
                {
                    "qid": "q1",
                    "question": "Name?",
                    "choices": None,
                    "multi_select": False,
                },
            ]
        },
        qids=["q0", "q1"],
    )
    first = _answer("2")
    assert first and first["result"]["status"] == "clarify_answered"
    assert not req.event.is_set(), "q1 is still open after locking q0"
    assert req.locked == {"q0": "prod"}
    second = _answer("blue db")
    assert second and second["result"]["status"] == "clarify_answered"
    assert req.event.is_set() and req.answered
    assert req.result["answers"] == {"q0": "prod", "q1": "blue db"}
    assert (
        req.result["outcome"] == "submitted"
    )  # the wire shape every card submission carries


def test_non_answers_fall_through():
    req = _open_clarify(
        "clarify-chat-sid", {"question": "Which?", "choices": ["a", "b"]}
    )
    assert _answer("/model") is None  # a slash command is a command, not an answer
    assert _answer("") is None
    assert _answer(None) is None
    assert (
        _answer("staging", turn_author={"id": "bot"}) is None
    )  # a bot-relay injection is not a chat reply
    assert not req.event.is_set()
    server_requests.cancel("clarify-chat-sid")
    assert _answer("a") is None  # no pending clarify → normal busy routing


def test_compute_host_mirror_settles_through_the_relay(monkeypatch):
    """A clarify minted by a compute-host child is mirrored on the session, not in the registry;
    the answer must travel back through the compute-host response relay."""
    session = _session()
    session["_compute_host_open_request"] = {
        "id": "srq-host123",
        "method": "clarify",
        "params": {
            "session_id": "clarify-chat-sid",
            "question": "Which?",
            "choices": ["a", "b"],
        },
    }
    relayed = []

    def _fake_relay(frame):
        relayed.append(frame)
        return True

    # The helper resolves through the server global the bridge publishes (bind_module), so the
    # stand-in replaces it there.
    monkeypatch.setattr(server, "_relay_compute_host_response", _fake_relay)
    resp = _answer("1", session=session)
    assert resp and resp["result"]["status"] == "clarify_answered"
    assert relayed == [
        {"jsonrpc": "2.0", "id": "srq-host123", "result": {"answer": "a"}}
    ]


def test_prompt_submit_answers_pending_clarify_before_busy_routing(monkeypatch):
    """End to end: a busy session with an open clarify request must resolve it from the typed
    prompt.submit instead of steering/queueing the reply behind the very wait it answers."""
    sid = "clarify-chat-sid"
    session = _session(running=True)
    server._sessions[sid] = session
    req = _open_clarify(sid, {"question": "Which?", "choices": ["a", "b"]})
    busy_calls = []
    monkeypatch.setattr(
        server,
        "_handle_busy_submit",
        lambda *a, **kw: busy_calls.append(a) or {"result": {"status": "queued"}},
    )
    try:
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "1"},
        })
        assert resp and resp.get("result", {}).get("status") == "clarify_answered", (
            f"got: {resp}"
        )
        assert not busy_calls, (
            "the clarify reply must not reach the busy steer/queue path"
        )
        assert req.event.is_set() and req.result == {"answer": "a"}
    finally:
        server._sessions.pop(sid, None)
