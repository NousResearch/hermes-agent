"""An interrupt that lands while ``prompt.submit`` is still setting up its turn ends that submit's turn.

``prompt.submit`` claims the turn (``running=True``) before it persists the prompt, starts the agent build and
publishes the dispatch thread.
In that window ``_run_thread`` is the previous turn's finished thread, so ``session.interrupt`` from a second
transport on the same session clears ``running``. A second ``prompt.submit`` can then claim the session again. The
first submit must not run its prompt beside the second one, and must leave the second turn's state alone: its
published worker, ``running``, the in-flight turn, the staged user row and the lease."""

import contextlib
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


@pytest.mark.parametrize(("block_in", "second_submit", "first_row_stored", "expected"), [
    pytest.param("persist", False, True, {
        "first": {"status": "streaming"},
        "ran": [],
        "errors": [{"message": "Turn cancelled before the agent was ready"}],
        "running": False,
        "inflight_user": None,
        "staged_user": None,
        "published_worker": None,
        "lease_released": False,
        "written": ["first"],
    }, id="interrupt-only"),
    pytest.param("persist", True, True, {"first": 4125, **_SECOND_TURN_KEPT, "written": ["second"]},
                 id="second-submit"),
    pytest.param("persist", True, False, {"first": 5072, **_SECOND_TURN_KEPT, "written": ["second"]},
                 id="second-submit-first-store-unavailable"),
    pytest.param("build", True, True, {"first": 4125, **_SECOND_TURN_KEPT, "written": ["first", "second"]},
                 id="second-submit-during-first-build"),
])
def test_an_interrupt_during_submit_setup_ends_that_submit(
    monkeypatch, block_in, second_submit, first_row_stored, expected
):
    sid = "interrupt-claim-sid"
    session = _idle_session()
    first_blocked, release_first = threading.Event(), threading.Event()
    second_published, finish_turns = threading.Event(), threading.Event()
    ran, errors, workers, released, written = [], [], {}, [], []

    def block_first(step):
        if step == block_in and not first_blocked.is_set():
            first_blocked.set()
            assert release_first.wait(10)
            return True
        return False

    def ensure_row(_session):
        return first_row_stored if block_first("persist") else True

    def write_row(_session, text, *_a):
        written.append(text)
        return {"role": "user", "content": text}

    def run_prompt_submit(_rid, _sid, sess, text, **_kw):
        ran.append(text)
        workers[text] = server._start_session_work(
            lambda: finish_turns.wait(10), name=f"prompt-turn-{text}", session=sess)
        second_published.set()

    monkeypatch.setattr(server, "_ensure_session_db_row", ensure_row)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_write_submit_user_row", write_row)
    monkeypatch.setattr(server, "_release_active_session_slot", lambda _session: released.append(True))
    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt_submit)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: block_first("build"))
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
        assert first_blocked.wait(10)
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
                "written": written,
            }
    finally:
        release_first.set()
        finish_turns.set()
        for worker in workers.values():
            worker.join(10)
        server._sessions.pop(sid, None)

    assert outcome == expected


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("claim_state", ["current", "stopped", "replaced"])
def test_finalized_reopen_follows_the_admitted_submit_claim(
    tmp_path, monkeypatch, isolated, claim_state
):
    from hermes_state import SessionDB
    from tui_gateway import methods_prompt, rpc_dispatch
    from tui_gateway.method_ctx import rebind

    db = SessionDB(tmp_path / "state.db")
    db.create_session("finalized", source="desktop")
    db.append_message("finalized", "user", "old ask", timestamp=100.0)
    db.append_message("finalized", "assistant", "old answer", timestamp=101.0)
    db.end_session("finalized", "agent_close")
    session = _idle_session()
    session["session_key"] = "finalized"
    sid = "reopen-claim-sid"
    dispatched = []
    original_claim = rebind(methods_prompt._lock_in_submit_turn, vars(server))
    handle_request = rebind(rpc_dispatch.handle_request, vars(server))

    def claim(*args):
        result = original_claim(*args)
        if claim_state == "replaced":
            original_claim(
                "replacement", sid, session, "replacement", {}, False, None, None, None
            )
        elif claim_state == "stopped":
            with session["history_lock"]:
                session["running"] = False
                session["_turn_cancel_requested"] = True
        return result

    def dispatch(_rid, _sid, _session, text, **_kwargs):
        dispatched.append(text)
        return {"result": {"status": "streaming", "turn_isolation": True}}

    monkeypatch.setattr(server, "_lock_in_submit_turn", claim)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(
        server, "_session_db", lambda _session: contextlib.nullcontext(db)
    )
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: isolated)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server, "_restart_completed_failed_agent_build", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", dispatch)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    server._sessions[sid] = session
    try:
        response = handle_request({
            "id": "first",
            "method": "prompt.submit",
            "params": {
                "session_id": sid,
                "text": "new ask",
            },
        })
        assert response is not None
        worker = session.get("_run_thread")
        if worker is not None:
            worker.join(10)
            assert not worker.is_alive()
        row = db.get_session("finalized")
        assert row is not None
        messages = [
            (message["role"], message["content"])
            for message in db.get_messages("finalized")
        ]
        outcome = {
            "response": response.get("result") or response["error"]["code"],
            "ended": row["ended_at"] is not None,
            "end_reason": row["end_reason"],
            "messages": messages,
            "isolated_dispatch": dispatched,
        }
        replaced = claim_state == "replaced"
        assert outcome == {
            "response": 4125
            if replaced
            else (
                {"status": "streaming", "turn_isolation": True}
                if isolated
                else {
                    "status": "streaming",
                    "user_row_id": db.latest_message_row_id("finalized", role="user"),
                }
            ),
            "ended": replaced,
            "end_reason": "agent_close" if replaced else None,
            "messages": [("user", "old ask"), ("assistant", "old answer")]
            + ([] if replaced or isolated else [("user", "new ask")]),
            "isolated_dispatch": [] if replaced or not isolated else ["new ask"],
        }
    finally:
        server._sessions.pop(sid, None)
        db.close()
