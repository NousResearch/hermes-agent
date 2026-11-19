"""Regression tests for #79509: a model switch on a compute-host (turn-isolation)
session must cross the process boundary and be applied by the CHILD's live agent.

On an isolated session the live agent lives in the compute-host child; the serving
process sees ``session["agent"] is None``. Before the fix, ``config.set model``:

- busy: stashed ``pending_model_switch`` for a turn thread whose
  ``_apply_pending_model_switch`` pops it and returns early server-side, so the
  pick was silently dropped;
- idle: built a SECOND agent inside the serving process and switched that copy,
  while every actual turn ran in the child on the old model.

The fix carries the stash in the turn frame (copied, not popped), the child adopts
it, its turn thread applies it, and ``_on_compute_host_turn_done`` clears the
server-side stash only after a successful isolated turn.
"""

from __future__ import annotations

import threading
import time

from tui_gateway import server


def _session(**extra):
    base = dict(
        agent=None, agent_ready=threading.Event(), session_key="s1-key", history=[],
        history_version=0, history_lock=threading.Lock(), running=False,
        attached_images=[], image_counter=0, cols=80, slash_worker=None,
        show_reasoning=False, tool_progress_mode="all", inflight_turn=None,
        created_at=time.time(), last_active=time.time())
    base.update(extra)
    return base


def test_config_set_model_defers_on_compute_host_session(monkeypatch):
    """Idle isolated session: config.set model must stash, not direct-apply."""
    session = _session(_compute_host_active=True)
    server._sessions["sid-cfgset"] = session
    build_calls = []
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: build_calls.append(a))
    try:
        resp = server.handle_request(
            {"id": "1", "method": "config.set",
             "params": {"session_id": "sid-cfgset", "key": "model", "value": "anthropic/claude-sonnet-4.6"}})
        assert resp["result"].get("deferred") is True
        assert session["pending_model_switch"]["raw"] == "anthropic/claude-sonnet-4.6"
        # The broken path built a second agent inside the server process.
        assert build_calls == []
    finally:
        server._sessions.pop("sid-cfgset", None)


def test_compute_host_turn_frame_carries_pending_switch():
    session = _session(pending_model_switch={"raw": "anthropic/claude-sonnet-4.6",
                                             "display_model": "anthropic/claude-sonnet-4.6"})
    frame = server._compute_host_turn_frame("r1", "sid", session, "hello")
    assert frame["pending_model_switch"]["raw"] == "anthropic/claude-sonnet-4.6"
    # Copied, not popped — the fail-open in-process path may still need it.
    assert session["pending_model_switch"]["raw"] == "anthropic/claude-sonnet-4.6"


def test_compute_host_turn_done_clears_pending_switch_on_success(monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    session = _session(running=True, pending_model_switch={"raw": "anthropic/claude-sonnet-4.6"})
    server._sessions["sid-done"] = session
    try:
        server._on_compute_host_turn_done(
            "r1", "sid-done", session, {"type": "turn.end", "session_info_emitted": True})
        assert "pending_model_switch" not in session
        # An errored turn must KEEP the stash for the fail-open path.
        session["pending_model_switch"] = {"raw": "anthropic/claude-sonnet-4.6"}
        session["running"] = True
        server._on_compute_host_turn_done(
            "r1", "sid-done", session,
            {"type": "turn.error", "message": "boom", "session_info_emitted": True})
        assert session["pending_model_switch"]["raw"] == "anthropic/claude-sonnet-4.6"
    finally:
        server._sessions.pop("sid-done", None)
