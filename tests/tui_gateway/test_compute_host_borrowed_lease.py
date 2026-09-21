"""Regression tests for #101416: isolated (compute-host) turns refused their own session.

With ``dashboard.turn_isolation: true`` every lazy (agent-not-yet-built, i.e. every NEW) desktop
session's turn is routed to the compute-host CHILD process. The parent claims the session's
active-session lease in ``prompt.submit`` before routing; the child's freshly built session record
carried NO lease, so ``_admit_prompt_turn`` re-claimed from the child's pid and was fenced out by the
parent's own registry entry (``_is_same_writer`` requires the same pid AND the same live_session_id).

The fix: the parent vouches on the turn frame (``active_session_lease`` = {lease_id, session_id}) and
the child installs an INERT borrow (``ActiveSessionLease(enabled=False)``) before the turn pipeline
runs. The REAL lease never leaves the parent: it is re-anchored there on a child-side compression
rotation and held past ``session.close`` until the child's turn settles.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import types

import pytest

from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


def _frames(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _wait(out: io.StringIO, predicate, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _frames(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out; saw={_frames(out)}")


def _stub_agent(deltas: list[str]) -> types.SimpleNamespace:
    def run_conversation(prompt, *, conversation_history=None, stream_callback=None, **_kw):
        final = "".join(deltas)
        if stream_callback is not None:
            for chunk in deltas:
                stream_callback(chunk)
        messages = [*(conversation_history or []), {"role": "user", "content": prompt},
                    {"role": "assistant", "content": final}]
        return {"final_response": final, "messages": messages}

    return types.SimpleNamespace(
        session_id="s1-key", run_conversation=run_conversation,
        clear_interrupt=lambda: None, hard_interrupt=lambda *a, **k: None)


def _make_frame(sid: str, **overrides) -> dict:
    frame = {"type": "turn.start", "sid": sid, "request_id": "turn", "text": "hello",
             "session_key": "s1-key", "source": "desktop", "cols": 80, "history": []}
    frame.update(overrides)
    return frame


def _seed_parent_lease(key: str, live_session_id: str = "parent-sid"):
    """Claim the lease exactly as the parent dashboard would (same pid, its own live id)."""
    from hermes_cli.active_sessions import try_acquire_active_session

    lease, message = try_acquire_active_session(
        session_id=key, surface="desktop", config={},
        metadata={"live_session_id": live_session_id}, track_liveness=True)
    assert message is None and lease is not None
    return lease


def _foreign_acquire(key: str):
    """A DISTINCT writer (same pid, another live id — the exact _is_same_writer fence)."""
    from hermes_cli.active_sessions import try_acquire_active_session

    return try_acquire_active_session(
        session_id=key, surface="cli", config={}, metadata={"live_session_id": "other-writer"})


def _registry() -> list[dict]:
    path = os.path.join(os.environ["HERMES_HOME"], "runtime", "active_sessions.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("entries", [])


def _parent_session(sid: str, key: str, lease) -> dict:
    return dict(agent=None, agent_ready=threading.Event(), session_key=key, history=[], history_version=0,
                history_lock=threading.Lock(), running=True, transport=server._detached_ws_transport,
                attached_images=[], cols=80, source="desktop", inflight_turn=None, created_at=time.time(),
                last_active=time.time(), active_session_lease=lease, _compute_host_active=True, _sid=sid)


@pytest.fixture()
def isolated_env(monkeypatch, tmp_path):
    """Real _build_server_session → _init_session → _run_prompt_submit → _admit_prompt_turn pipeline,
    with the environment-heavy side paths neutralized. The turn BODY is cut right after admission
    (``_prepare_turn_input`` → None), so the lease path under test runs REAL against the
    conftest-sandboxed HERMES_HOME registry while nothing calls a provider."""
    agent = _stub_agent(["a ", "b "])
    monkeypatch.setattr(server, "_make_agent", lambda *a, **kw: agent)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent_: {})
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_session_services", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    import tui_gateway.prompt_turn as prompt_turn

    for mod in (server, prompt_turn):
        if hasattr(mod, "_prepare_turn_input"):
            monkeypatch.setattr(mod, "_prepare_turn_input", lambda *a, **k: None)
    yield agent
    for sid in [s for s in list(server._sessions) if s.startswith("s1")]:
        server._sessions.pop(sid, None)


def _run_turn(frame: dict, timeout: float = 5.0) -> tuple[list[dict], dict | None]:
    """Run one turn.start through the real child path; return (all frames, turn.end frame)."""
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    try:
        host.handle_frame(frame)
        end = _wait(out, lambda f: f["type"] == "turn.end", timeout=timeout)
    finally:
        host.close()
    return _frames(out), end


# ── Fix 1: the child borrows instead of re-claiming ─────────────────────────


def test_isolated_turn_runs_against_parent_leased_session(isolated_env):
    """THE #101416 repro, fixed: parent holds the lease, child runs the turn end-to-end and the
    registry still holds exactly the parent's entry."""
    parent = _seed_parent_lease("s1-key")
    try:
        frames, end = _run_turn(_make_frame(
            "s1", active_session_lease={"lease_id": parent.lease_id, "session_id": "s1-key"}))
        kinds = [f["type"] for f in frames]
        assert "turn.started" in kinds and kinds[-1] == "turn.end" and end["session_key"] == "s1-key"
        events = [(f["message"].get("params") or {}).get("type") for f in frames if f["type"] == "rpc"]
        assert "error" not in events and "message.start" in events  # admitted and ran, no refusal
        entries = _registry()
        assert [e["lease_id"] for e in entries] == [parent.lease_id]
        assert entries[0]["metadata"]["live_session_id"] == "parent-sid"
    finally:
        parent.release()


def test_isolated_turn_without_matching_vouch_still_fails_closed(isolated_env):
    """Negative control: a vouch for another stored id (a lease still keyed on the pre-rotation id)
    keeps the legacy self-claim, which the parent's entry still fences — nothing about
    _is_same_writer is relaxed, and a stale lease never authorizes the continuation."""
    parent = _seed_parent_lease("s1-key")
    try:
        frames, _ = _run_turn(_make_frame(
            "s1", active_session_lease={"lease_id": parent.lease_id, "session_id": "stale-A"}))
        errors = [f["message"]["params"]["payload"]["message"] for f in frames
                  if f["type"] == "rpc" and (f["message"].get("params") or {}).get("type") == "error"]
        assert errors and "open in another Hermes window" in errors[0]
        assert [e["lease_id"] for e in _registry()] == [parent.lease_id]
    finally:
        parent.release()


# ── Fix 2: compression rotation A->B stays owned by the parent ──────────────


def test_child_rotation_never_claims_and_parent_reanchors_its_real_lease():
    parent = _seed_parent_lease("A")
    try:
        # Child side: the borrow is retargeted locally; the registry is untouched (no child-pid lease).
        child = {"session_key": "A", "history_lock": threading.Lock()}
        server._install_borrowed_lease("sid", child, _make_frame(
            "sid", session_key="A", active_session_lease={"lease_id": parent.lease_id, "session_id": "A"}))
        assert server._transfer_active_session_slot("sid", child, new_session_id="B") is True
        assert child["active_session_lease"].session_id == "B"
        assert [(e["session_id"], e["pid"]) for e in _registry()] == [("A", os.getpid())]
        # Parent side: a stale lease vouches for nothing; adopting the rotated key moves the REAL lease.
        session = _parent_session("sid", "A", parent)
        session["session_key"] = "B"
        assert server._active_session_lease_vouch(session) is None
        session["session_key"] = "A"
        with session["history_lock"]:
            server._compute_host_adopt_frame_meta(session, {"sid": "sid", "session_key": "B"})
        assert session["session_key"] == "B" and parent.session_id == "B"
        assert [(e["session_id"], e["lease_id"]) for e in _registry()] == [("B", parent.lease_id)]
        assert server._active_session_lease_vouch(session) == {"lease_id": parent.lease_id, "session_id": "B"}
    finally:
        parent.release()


# ── Fix 3: close keeps the lease until the isolated turn settles ────────────


def test_close_holds_lease_until_isolated_turn_settles(monkeypatch):
    interrupts: list[str] = []
    monkeypatch.setattr(server, "_get_compute_host_supervisor",
                        lambda *a, **k: types.SimpleNamespace(interrupt=lambda sid, **k: interrupts.append(sid)))
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda *a: {"turn_isolation": True})
    monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.2)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    parent = _seed_parent_lease("A")
    session = _parent_session("sid", "A", parent)
    session["_compute_host_turn_id"] = "turn-1"  # the child is still running this turn
    session["_closing"] = True
    try:
        assert server._teardown_popped_session(session, end_reason="tui_close") is True
        assert interrupts == ["sid"]
        # Close returned, the child is live: ownership is still ours and still refuses a distinct writer.
        assert [e["lease_id"] for e in _registry()] == [parent.lease_id]
        assert parent.lease_id in server._own_live_lease_ids()
        lease, refusal = _foreign_acquire("A")
        assert lease is None and getattr(refusal, "reason", "") == "SESSION_NOT_OWNED"
        # Child settlement (turn.end, or turn.error from _fail_pending_turns on child death) releases it.
        server._on_compute_host_turn_done("rid", "sid", session, {"type": "turn.end", "sid": "sid", "session_key": "A"})
        assert _registry() == [] and parent.lease_id not in server._own_live_lease_ids()
        lease, refusal = _foreign_acquire("A")
        assert refusal is None and lease is not None
        lease.release()
    finally:
        parent.release()
