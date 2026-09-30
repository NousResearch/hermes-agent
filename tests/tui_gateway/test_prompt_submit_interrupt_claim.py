"""An interrupt that lands while ``prompt.submit`` is still persisting its prompt ends that submit's turn.

``prompt.submit`` claims the turn (``running=True``) before it persists the prompt and starts the dispatch thread.
In that window ``_run_thread`` is the previous turn's finished thread, so ``session.interrupt`` from a second
transport on the same session clears ``running``. A second ``prompt.submit`` can then claim the session again, and
the first submit's dispatch thread must not run its prompt beside the second one."""

import threading
import types

import pytest

from tui_gateway import server


def _idle_session() -> dict:
    ready = threading.Event()
    ready.set()
    return {
        "agent": types.SimpleNamespace(),
        "agent_ready": ready,
        "session_key": "interrupt-claim-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
    }


def _turn_threads(session: dict) -> set[threading.Thread]:
    thread = session.get("_run_thread")
    return {thread} if isinstance(thread, threading.Thread) else set()


@pytest.mark.parametrize("second_submit", [False, True])
def test_an_interrupt_during_submit_persist_ends_that_submit(monkeypatch, second_submit):
    sid = "interrupt-claim-sid"
    session = _idle_session()
    first_persisting = threading.Event()
    release_first = threading.Event()
    ran, errors = [], []

    def persist(_rid, _session, text=None, _display_kind=None):
        if text == "first":
            first_persisting.set()
            assert release_first.wait(10)

    monkeypatch.setattr(server, "_persist_session_row_for_submit", persist)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda _rid, _sid, _session, text, **_kw: ran.append(text))
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server, "_emit", lambda event, _sid, payload=None, *_a, **_k: errors.append(payload) if event == "error" else None)
    server._sessions[sid] = session
    responses, threads = {}, set()

    def submit(text):
        responses[text] = server.handle_request(
            {"id": text, "method": "prompt.submit", "params": {"session_id": sid, "text": text}})

    first = threading.Thread(target=submit, args=("first",))
    try:
        first.start()
        assert first_persisting.wait(10)
        responses["interrupt"] = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}})
        if second_submit:
            submit("second")
            threads |= _turn_threads(session)
        release_first.set()
        first.join(10)
        threads |= _turn_threads(session)
        for thread in threads:
            thread.join(10)

        outcome = {
            "responses": {rid: resp.get("result") for rid, resp in responses.items()},
            "ran": ran,
            "errors": errors,
            "running": session["running"],
            "inflight_user": (session.get("inflight_turn") or {}).get("user"),
        }
    finally:
        release_first.set()
        server._sessions.pop(sid, None)

    if second_submit:
        assert outcome == {
            "responses": {
                "first": {"status": "streaming"},
                "interrupt": {"status": "interrupted"},
                "second": {"status": "streaming"},
            },
            "ran": ["second"],
            "errors": [],
            "running": True,
            "inflight_user": "second",
        }
    else:
        assert outcome == {
            "responses": {"first": {"status": "streaming"}, "interrupt": {"status": "interrupted"}},
            "ran": [],
            "errors": [{"message": "Turn cancelled before the agent was ready"}],
            "running": False,
            "inflight_user": None,
        }
