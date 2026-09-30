"""An interrupt that lands while ``prompt.submit`` is still persisting its prompt ends that submit's turn.

``prompt.submit`` claims the turn (``running=True``) before it persists the prompt and publishes the dispatch thread.
In that window ``_run_thread`` is the previous turn's finished thread, so ``session.interrupt`` from a second
transport on the same session clears ``running``. A second ``prompt.submit`` can then claim the session again. The
first submit must not run its prompt beside the second one, and must leave the second turn's state alone: its
published worker, ``running``, the in-flight turn, the staged user row and the lease."""

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


_SECOND_TURN_KEPT = {
    "second": {"status": "streaming"},
    "ran": ["second"],
    "errors": [],
    "running": True,
    "inflight_user": "second",
    "staged_user": "second",
    "published_worker": "second",
    "lease_released": False,
}


@pytest.mark.parametrize(("second_submit", "first_row_stored", "expected"), [
    pytest.param(False, True, {
        "first": {"status": "streaming"},
        "ran": [],
        "errors": [{"message": "Turn cancelled before the agent was ready"}],
        "running": False,
        "inflight_user": None,
        "staged_user": None,
        "published_worker": None,
        "lease_released": False,
    }, id="interrupt-only"),
    pytest.param(True, True, {"first": 4125, **_SECOND_TURN_KEPT}, id="second-submit"),
    pytest.param(True, False, {"first": 5072, **_SECOND_TURN_KEPT}, id="second-submit-first-store-unavailable"),
])
def test_an_interrupt_during_submit_persist_ends_that_submit(monkeypatch, second_submit, first_row_stored, expected):
    sid = "interrupt-claim-sid"
    session = _idle_session()
    first_persisting, release_first = threading.Event(), threading.Event()
    second_published, finish_turns = threading.Event(), threading.Event()
    ran, errors, workers, released = [], [], {}, []

    def ensure_row(_session):
        if not first_persisting.is_set():
            first_persisting.set()
            assert release_first.wait(10)
            return first_row_stored
        return True

    def run_prompt_submit(_rid, _sid, sess, text, **_kw):
        ran.append(text)
        workers[text] = server._start_session_work(
            lambda: finish_turns.wait(10), name=f"prompt-turn-{text}", session=sess)
        second_published.set()

    monkeypatch.setattr(server, "_ensure_session_db_row", ensure_row)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(
        server, "_write_submit_user_row", lambda _session, text, *_a: {"role": "user", "content": text})
    monkeypatch.setattr(server, "_release_active_session_slot", lambda _session: released.append(True))
    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt_submit)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server, "_emit", lambda event, _sid, payload=None, *_a, **_k: errors.append(payload) if event == "error" else None)
    server._sessions[sid] = session
    responses = {}

    def submit(text):
        resp = server.handle_request({"id": text, "method": "prompt.submit", "params": {"session_id": sid, "text": text}})
        responses[text] = resp.get("result") or resp["error"]["code"]

    first = threading.Thread(target=submit, args=("first",))
    try:
        first.start()
        assert first_persisting.wait(10)
        interrupted = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}})
        assert interrupted["result"] == {"status": "interrupted"}
        if second_submit:
            submit("second")
            assert second_published.wait(10)
        release_first.set()
        first.join(10)
        published = session.get("_run_thread")
        if published is not None and published not in workers.values():
            published.join(10)

        with session["history_lock"]:
            outcome = {
                **responses,
                "ran": ran,
                "errors": errors,
                "running": session["running"],
                "inflight_user": (session.get("inflight_turn") or {}).get("user"),
                "staged_user": (session.get("_submit_user_row") or {}).get("content"),
                "published_worker": next((text for text, w in workers.items() if w is published), None),
                "lease_released": bool(released),
            }
    finally:
        release_first.set()
        finish_turns.set()
        for worker in workers.values():
            worker.join(10)
        server._sessions.pop(sid, None)

    assert outcome == expected
