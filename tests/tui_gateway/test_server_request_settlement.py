"""A terminal reply owns a server request even before its waiting thread resumes."""

import pytest

from tui_gateway import server_requests


@pytest.fixture(autouse=True)
def isolated_requests():
    original_write = server_requests._write
    original_emit = server_requests._emit
    server_requests.reset_for_tests()
    yield
    server_requests.reset_for_tests()
    server_requests.bind_sinks(original_write, original_emit)


@pytest.mark.parametrize("terminal", ["result", "error", "cancel"])
def test_terminal_reply_cannot_be_replaced_before_waiter_resumes(terminal):
    observations = {}
    first_answer = {"answer": "first answer"}

    def peer(frame):
        request_id = frame["id"]
        if terminal == "cancel":
            observations["settled"] = server_requests.cancel("session", "interrupted")
        else:
            reply = {"result": first_answer} if terminal == "result" else {
                "error": {"code": -32601, "message": "unsupported request"}}
            observations["settled"] = server_requests.resolve_response({"id": request_id, **reply})
        # A second renderer or replay can answer before the agent thread wakes.
        observations["replay"] = server_requests.open_requests("session")
        observations["pending"] = server_requests.pending_kind("session")
        observations["duplicate"] = server_requests.resolve_response(
            {"id": request_id, "result": {"answer": "late answer"}})
        observations["cancelled"] = server_requests.cancel("session")

    server_requests.bind_sinks(peer, lambda *_: None)
    answer = server_requests.send("clarify", "session", {"question": "Choose an answer"}, timeout=1)

    assert answer == (first_answer if terminal == "result" else None)
    assert observations == {
        "settled": True, "replay": [], "pending": "", "duplicate": False, "cancelled": 0}


def test_last_batch_lock_makes_the_answer_set_terminal():
    observations = {}
    locked_answers = {"first": "first answer", "second": "second answer"}

    def peer(frame):
        request_id = frame["id"]
        observations["remaining"] = server_requests.lock_answer(request_id, "first", locked_answers["first"])
        observations["partial"] = server_requests.open_requests("session")[0]["params"]["answers"]
        observations["complete"] = server_requests.lock_answer(request_id, "second", locked_answers["second"])
        observations["replay"] = server_requests.open_requests("session")
        observations["late_lock"] = server_requests.lock_answer(request_id, "first", "changed answer")
        observations["duplicate"] = server_requests.resolve_response(
            {"id": request_id, "result": {"answers": {"first": "late answer"}}})
        observations["cancelled"] = server_requests.cancel("session")

    server_requests.bind_sinks(peer, lambda *_: None)
    answer = server_requests.send(
        "clarify", "session",
        {"questions": [{"qid": qid, "question": qid} for qid in locked_answers]},
        qids=list(locked_answers), timeout=1)

    assert answer == {"answers": locked_answers}
    assert observations == {
        "remaining": ["second"], "partial": {"first": locked_answers["first"]},
        "complete": [], "replay": [], "late_lock": None, "duplicate": False, "cancelled": 0}
