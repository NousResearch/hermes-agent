"""Mounting a finalized session is read-only (#85303): session.resume / its deferred
hydration / the lazy watch path must NOT clear ``ended_at``/``end_reason`` — only the
first real turn (prompt.submit) reopens the row, so DB-derived liveness cannot re-light
a finished session just because someone opened it."""

import contextlib
import io
import json
import threading
import time
import types

import pytest

from hermes_state import SessionDB
from tui_gateway import server


def _frames(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _wait_for(out: io.StringIO, predicate, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _frames(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out; saw={_frames(out)}")


def _finalized_db(tmp_path, *, ended_reason="agent_close"):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(home / "state.db")
    db.create_session("finalized", source="desktop")
    db.append_message("finalized", "user", "old ask", timestamp=100.0)
    db.append_message("finalized", "assistant", "old answer", timestamp=101.0)
    db.end_session("finalized", ended_reason)
    return db, home


def _mount(monkeypatch, db, home, tmp_path, *, defer_history=False):
    events = []
    built = threading.Event()
    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None, **kwargs: db)
    monkeypatch.setattr(server, "_profile_home", lambda p: home if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: built.set())
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, payload)))
    return events, built


@pytest.mark.parametrize("defer_history", [False, True])
def test_resume_mount_keeps_finalized_row_ended(tmp_path, monkeypatch, defer_history):
    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "finalized", "source": "desktop", "defer_history": defer_history,
        }})
        assert response is not None and "error" not in response, response
        sid = response["result"]["session_id"]
        if defer_history:
            session = server._sessions[sid]
            assert session["resume_history_ready"].wait(5)
        else:
            assert built.wait(5)
        row = db.get_session("finalized")
        assert row["ended_at"] is not None, "mounting a finalized session must not reopen its row (#85303)"
        assert row["end_reason"] == "agent_close"
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


def test_lazy_watch_mount_keeps_finalized_row_ended(tmp_path, monkeypatch):
    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "finalized", "source": "desktop", "lazy": True,
        }})
        assert response is not None and "error" not in response, response
        row = db.get_session("finalized")
        assert row["ended_at"] is not None, "the lazy watch mount must not reopen a finalized row"
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


def test_first_real_turn_reopens_a_finalized_row(tmp_path, monkeypatch):
    """The activity gate: prompt.submit (a real send) is what clears ended_at, not the mount."""
    from tui_gateway import methods_prompt

    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    reopened = []
    monkeypatch.setattr(db, "reopen_session", lambda sid: reopened.append(sid) or
                        SessionDB.reopen_session(db, sid))
    reopened_row = db.get_session("finalized")
    assert reopened_row["ended_at"] is not None
    # The submit path's reopen helper: the row is finalized, so a real send reopens it.
    methods_prompt._reopen_if_finalized(db, "finalized")
    assert reopened == ["finalized"]
    row = db.get_session("finalized")
    assert row["ended_at"] is None, "the first real turn must reopen the finalized row"
    db.close()


def test_reopen_if_finalized_leaves_live_rows_untouched(tmp_path):
    db, home = _finalized_db(tmp_path)
    from tui_gateway import methods_prompt

    db.create_session("live", source="desktop")
    db.append_message("live", "user", "ask", timestamp=100.0)
    methods_prompt._reopen_if_finalized(db, "live")
    methods_prompt._reopen_if_finalized(db, "missing-entirely")
    row = db.get_session("live")
    assert row["ended_at"] is None and row["end_reason"] is None
    db.close()


def test_isolated_dispatch_reopens_a_finalized_row(tmp_path, monkeypatch):
    """The compute-host seam: prompt.submit returns BEFORE its persist on an isolated turn
    (the child process runs the turn), so the reopen must live on the dispatch path every
    turn crosses, not in the skipped persist. A finalized row a lazy mount is about to send
    through must be reopened before the child writes into it."""
    from tui_gateway import server

    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session, cfg=None: True)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    dispatched = {}
    monkeypatch.setattr(server, "_submit_prompt_to_compute_host",
                        lambda rid, sid, session, text, **kw: dispatched.update(sid=sid, text=text)
                        or {"result": {"status": "streaming", "turn_isolation": True}})
    monkeypatch.setattr(server, "_handle_busy_submit", lambda *a, **k: None)
    sid = None
    try:
        # Mount first (read-only): the submit then routes through the mounted session record.
        mounted = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "finalized", "source": "desktop", "lazy": True}})
        assert mounted is not None and "error" not in mounted, mounted
        sid = mounted["result"]["session_id"]
        session = server._sessions[sid]
        session["agent"] = types.SimpleNamespace(session_id="finalized", clear_interrupt=lambda: None)
        session["agent_ready"] = threading.Event()
        response = server.handle_request({"id": "submit", "method": "prompt.submit", "params": {
            "session_id": sid, "text": "one more thing"}})
        assert response is not None and "error" not in response, response
        assert dispatched.get("text") == "one more thing"
        row = db.get_session("finalized")
        assert row["ended_at"] is None, (
            "an isolated (compute-host) dispatch is a real turn: the finalized row must be "
            "reopened even though prompt.submit returns before its own persist (#85303)")
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


def test_compute_host_child_turn_reopens_a_finalized_row(tmp_path, monkeypatch):
    """The child-process half: the compute host builds its own session record and runs
    ``_run_prompt_submit`` directly, so the reopen at the dispatch seam fires there too —
    the turn's transcript is written into a live row even though the child never saw
    prompt.submit."""
    from tui_gateway import server
    from tui_gateway.compute_host import ComputeHost

    db, home = _finalized_db(tmp_path)
    events, built = _mount(monkeypatch, db, home, tmp_path)
    # The child-side session record: built from a turn.start frame, agent already attached.
    agent = types.SimpleNamespace(session_id="finalized", clear_interrupt=lambda: None)
    session = {
        "agent": agent, "session_key": "finalized", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0,
        "cols": 80, "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
        "inflight_turn": None, "created_at": time.time(), "last_active": time.time(),
    }
    server._sessions["child-sid"] = session
    # Cut the turn right after admission (the reopen runs BEFORE admission).
    from tui_gateway import prompt_turn
    monkeypatch.setattr(server, "_prepare_turn_input", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(prompt_turn, "_prepare_turn_input", lambda *a, **k: None, raising=False)
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "turn.start", "sid": "child-sid", "request_id": "t",
                           "text": "hello", "session_key": "finalized", "source": "desktop"})
        _wait_for(out, lambda f: f["type"] == "turn.end", timeout=10.0)
        row = db.get_session("finalized")
        assert row["ended_at"] is None, (
            "a turn run by the compute-host child is a real turn: the finalized row must be "
            "reopened before the child's transcript writes land (#85303)")
    finally:
        server._sessions.pop("child-sid", None)
        host.close()
        db.close()


@pytest.mark.parametrize("isolated", [False, True], ids=["inline", "isolated"])
@pytest.mark.parametrize("claim_state", ["current", "stopped", "replaced"])
def test_submit_reopens_only_for_its_owned_claim(
    tmp_path, monkeypatch, isolated, claim_state
):
    db, home = _finalized_db(tmp_path)
    _mount(monkeypatch, db, home, tmp_path)
    monkeypatch.setattr(
        server,
        "_session_uses_compute_host",
        lambda session, cfg=None: isolated if cfg else False,
    )
    monkeypatch.setattr(
        server,
        "_load_dashboard_process_isolation_config",
        lambda: {"turn_isolation": isolated},
    )
    monkeypatch.setattr(server, "_start_turn_thread", lambda *args: True)
    dispatched = []

    def submit_turn(frame, on_complete):
        dispatched.append(frame["text"])

    monkeypatch.setattr(
        server,
        "_get_compute_host_supervisor",
        lambda *args: types.SimpleNamespace(submit_turn=submit_turn),
    )
    monkeypatch.setattr(
        server,
        "_compute_host_turn_frame",
        lambda rid, sid, session, text, **kwargs: {"sid": sid, "text": text},
    )
    admission = server._session_turn_admission
    sid = None
    try:
        mounted = server.handle_request({
            "id": "resume",
            "method": "session.resume",
            "params": {"session_id": "finalized", "source": "desktop", "lazy": True},
        })
        assert mounted is not None and "error" not in mounted, mounted
        sid = mounted["result"]["session_id"]
        session = server._sessions[sid]
        session["agent"] = types.SimpleNamespace(
            session_id="finalized", clear_interrupt=lambda: None
        )
        session["agent_ready"] = threading.Event()

        @contextlib.contextmanager
        def stop_after_admission(current):
            with admission(current) as admitted:
                yield admitted
            if claim_state == "current":
                return
            interrupted = server.handle_request({
                "id": "stop",
                "method": "session.interrupt",
                "params": {"session_id": sid},
            })
            assert interrupted is not None and "error" not in interrupted, interrupted
            if claim_state == "replaced":
                with current["history_lock"]:
                    server._claim_session_turn(current)

        monkeypatch.setattr(server, "_session_turn_admission", stop_after_admission)
        response = server.handle_request({
            "id": "submit",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "one more thing"},
        })
        assert response is not None, response
        row = db.get_session("finalized")
        observed = {
            "finalized": row["ended_at"] is not None,
            "end_reason": row["end_reason"],
            "messages": [
                (message["role"], message["content"])
                for message in db.get_messages("finalized")
            ],
            "dispatched": dispatched,
            "error_code": (response.get("error") or {}).get("code"),
            "claim": session["_turn_claim"],
            "running": session["running"],
        }
        accepted = claim_state != "replaced"
        isolated_dispatch = isolated and claim_state == "current"
        assert observed == {
            "finalized": not accepted,
            "end_reason": None if accepted else "agent_close",
            "messages": [("user", "old ask"), ("assistant", "old answer")]
            + (
                [("user", "one more thing")]
                if accepted and not isolated_dispatch
                else []
            ),
            "dispatched": ["one more thing"] if isolated_dispatch else [],
            "error_code": None if accepted else 4125,
            "claim": 2 if claim_state == "replaced" else 1,
            "running": claim_state != "stopped",
        }
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()
