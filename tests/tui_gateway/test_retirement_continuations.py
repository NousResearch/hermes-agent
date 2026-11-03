"""Retirement cannot splice between a turn's completion and its automatic successors."""

import threading
import types


def test_crash_continuation_reserves_before_build_and_refuses_prepared_backend(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement
    from tui_gateway import server
    from tui_gateway.turn_marker import record_turn_start

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(server, "_auto_continue_config", lambda: (True, 3600, 3))
    session = {"profile_home": str(tmp_path), "history_lock": threading.RLock(), "running": False}
    record_turn_start(tmp_path, "recovery", "unfinished")
    building, release = threading.Event(), threading.Event()

    def build(*args):
        building.set()
        assert release.wait(10)

    monkeypatch.setattr(server, "_start_agent_build", build)
    monkeypatch.setattr(server, "_wait_agent", lambda *args, **kw: {"error": "no model in test"})
    token = fence.prepare()["token"]
    try:
        assert server._maybe_schedule_auto_continue("s", session, "recovery") is None
        assert not building.is_set()
        assert fence.cancel(token) == {"ok": True}
        assert server._maybe_schedule_auto_continue("s", session, "recovery") is not None
        assert building.wait(10)
        assert not session["running"]
        assert fence.prepare() == {"ok": False, "idle": False}
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "auto-continue-s":
                thread.join(10)
    assert fence.prepare()["ok"] is True


def test_prompt_worker_stays_busy_after_running_flag_until_finalizers_finish(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement
    from tui_gateway import server

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    agent = types.SimpleNamespace(session_id="retirement-turn", clear_interrupt=lambda: None,
                                  run_conversation=lambda *args, **kwargs: {"final_response": "done"})
    session = {"agent": agent, "session_key": "retirement-turn", "history": [], "profile_home": str(tmp_path),
               "history_lock": threading.RLock(), "history_version": 0, "running": True, "attached_images": [],
               "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
               "tool_progress_mode": "all", "inflight_turn": None}
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd", "_sync_session_key_after_compress"):
        monkeypatch.setattr(server, name, lambda *args, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    finishing, release = threading.Event(), threading.Event()

    def finish(*args):
        finishing.set()
        assert release.wait(10)

    monkeypatch.setattr(server, "_emit_settled_session_info", finish)
    monkeypatch.setitem(server._sessions, "s", session)
    try:
        assert server._run_prompt_submit("r", "s", session, "hello") is True
        assert finishing.wait(10)
        assert session["running"] is False
        assert fence.prepare() == {"ok": False, "idle": False}
    finally:
        release.set()
        if thread := session.get("_run_thread"):
            thread.join(10)
    assert fence.prepare()["ok"] is True
