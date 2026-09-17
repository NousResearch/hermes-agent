"""Probe: superseded finalizer rewrites a LIVE newer turn's cwd mid-reconcile."""

from __future__ import annotations

import threading
import types

from tui_gateway import server as srv


def test_probe_superseded_finalizer_rewrites_live_newer_turn_cwd(monkeypatch, tmp_path):
    import tools.terminal_tool as terminal_tool
    from tui_gateway import git_probe

    lock = threading.RLock()
    new_repo = tmp_path / "wt-new"
    landed = tmp_path / "wt-landed"
    common = tmp_path / "common"
    for p in (new_repo, landed, common):
        p.mkdir()

    agent_obj = types.SimpleNamespace(session_id="a", session_cwd=str(new_repo))
    session = {
        "agent": agent_obj,
        "session_key": "k",
        "history": [],
        "history_lock": lock,
        "running": False,
        "cols": 80,
        "cwd": str(new_repo),
        "explicit_cwd": True,
        "cwd_from_settle": True,
        "_active_turn_id": "old-turn",
    }

    in_reconcile = threading.Event()
    resume = threading.Event()

    def parked_get_session_cwd(_key):
        # We are inside _reconcile_session_cwd_from_terminal, between the old
        # finalizer's two superseded() checks and NOT holding history_lock.
        in_reconcile.set()
        assert resume.wait(10)
        return str(landed)

    monkeypatch.setattr(srv, "_is_local_terminal_backend", lambda: True)
    monkeypatch.setattr(terminal_tool, "get_session_cwd", parked_get_session_cwd)
    monkeypatch.setattr(git_probe, "repo_root", lambda p: str(p))
    monkeypatch.setattr(git_probe, "common_repo_root", lambda p: str(common))
    monkeypatch.setattr(
        srv, "_persist_session_cwd_and_schedule_git_meta", lambda *a, **kw: None)
    registered = []
    real_register = srv._register_session_cwd
    monkeypatch.setattr(
        srv, "_register_session_cwd",
        lambda s: registered.append(s.get("cwd")) or real_register(s))
    emitted = []
    monkeypatch.setattr(srv, "_emit", lambda *a, **kw: emitted.append(a))

    result = {}

    def old_finalizer():
        result["published"] = srv._emit_settled_session_info(
            "sid", session, agent_obj, "old-turn")

    t = threading.Thread(target=old_finalizer, name="old-finalizer")
    t.start()
    assert in_reconcile.wait(10)

    # A NEWER turn wins admission and is live, with its own workspace.
    with lock:
        session["running"] = True
        session["_active_turn_id"] = "new-turn"
        session["cwd"] = str(new_repo)

    resume.set()
    t.join(10)

    print("published (skipped emit) =", result["published"])
    print("session.info emits       =", emitted)
    print("LIVE turn cwd after      =", session["cwd"])
    print("explicit_cwd/from_settle =",
          session.get("explicit_cwd"), session.get("cwd_from_settle"))
    print("agent.session_cwd        =", agent_obj.session_cwd)
    print("_register_session_cwd on =", registered)
    assert session["cwd"] == str(new_repo), (
        "a superseded turn rewrote the LIVE newer turn's cwd to "
        f"{session['cwd']!r} (and pushed it into agent.session_cwd / terminal "
        "task env) even though its session.info emit was correctly suppressed")
