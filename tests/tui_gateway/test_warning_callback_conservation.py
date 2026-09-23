"""Real callback and queue boundaries; renderer preference never hides operator logs."""
import logging
import queue
from types import SimpleNamespace

import pytest
import yaml

from agent.status_output import StatusOutputMixin
from tui_gateway import server
from tests.tui_gateway.test_auto_continue import turn_env, marker_home, _session


class Agent(StatusOutputMixin):
    log_prefix = ""
    suppress_status_output = True
    platform = "tui"
    _print_fn = None

    def _touch_activity(self, *args):
        pass


@pytest.mark.parametrize("setting", [None, False, True])
def test_real_tui_callbacks_filter_only_diagnostics_at_owner_sink(tmp_path, monkeypatch, setting):
    launch, owner = tmp_path / "launch", tmp_path / "owner"
    launch.mkdir()
    owner.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    (launch / "config.yaml").write_text(yaml.safe_dump({"display": {"suppress_warning_notifications": setting is not True}}))
    (owner / "config.yaml").write_text(yaml.safe_dump({} if setting is None else {"display": {"suppress_warning_notifications": setting}}))
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    for home, muted in [(owner, setting is True), (launch, setting is not True), (owner, setting is True)]:
        session = _session(profile_home=str(home))
        monkeypatch.setitem(server._sessions, "warning-callback", session)
        agent = Agent()
        for key, callback in server._agent_cbs("warning-callback").items():
            setattr(agent, key, callback)
        # Intentionally invoke outside the owning context, like a callback worker.
        frames.clear()
        agent._emit_warning("unfamiliar warning wording")
        agent._emit_diagnostic_status("unfamiliar fallback wording")
        agent._emit_diagnostic_wait("unfamiliar retry wording")
        agent._emit_notice(SimpleNamespace(level="error", text="notice detail", kind="custom", ttl_ms=0, key="custom", id="n"))
        diagnostic_frames = list(frames)
        agent._emit_status("⚠ user-requested progress with warning punctuation")
        assert bool(diagnostic_frames) is not muted
        assert len(diagnostic_frames) == (0 if muted else 4)
        assert frames[-1]["params"]["payload"]["text"] == "⚠ user-requested progress with warning punctuation"


@pytest.mark.parametrize("setting", [None, False, True])
def test_post_turn_drain_uses_owner_and_preserves_real_clarify(turn_env, marker_home, monkeypatch, setting, caplog):
    from tools.process_registry import process_registry
    from tui_gateway import server_requests
    owner = marker_home / "owner"
    owner.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text(yaml.safe_dump({"display": {"suppress_warning_notifications": setting is not True}}))
    (owner / "config.yaml").write_text(yaml.safe_dump({} if setting is None else {"display": {"suppress_warning_notifications": setting}}))
    frames, work, controls = [], [], []
    def wire(frame):
        frames.append(frame)
        if frame.get("method") == "clarify":
            assert server_requests.resolve_response({"id": frame["id"], "result": {"answer": "yes"}})
        return True
    monkeypatch.setattr(server, "write_json", wire)
    monkeypatch.setattr(server_requests, "_write", wire)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (server.threading.Event(), SimpleNamespace(join=lambda: None)))
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None)
    agent.clarify_callback = server._agent_cbs("drain-owner")["clarify_callback"]
    def run(message, **kwargs):
        work.append(message)
        controls.append(agent.clarify_callback("Continue recovery?", ["yes", "no"]))
        logging.getLogger("operator").warning("watch diagnostic retained")
        return {"final_response": "diagnostic final", "messages": []}
    agent.run_conversation = run
    session = _session(agent=agent, profile_home=str(owner))
    monkeypatch.setitem(server._sessions, "drain-owner", session)
    process_registry.completion_queue.put({"type": "watch_disabled", "session_id": "proc", "session_key": "session-key",
        "origin_ui_session_id": "drain-owner", "message": "watch diagnostic"})
    with caplog.at_level(logging.WARNING, logger="operator"):
        server._run_post_turn_followups("request", "drain-owner", session, {}, None)
    assert len(work) == 1
    assert controls == ["yes"]
    assert process_registry.completion_queue.empty()
    assert "watch diagnostic retained" in caplog.text
    presentation = [f for f in frames if f.get("params", {}).get("type") in {"status.update", "message.complete", "message.start", "error"}]
    assert bool(presentation) is not (setting is True)
    assert len([f for f in frames if f.get("method") == "clarify"]) == 1


@pytest.mark.parametrize("setting", [None, False, True])
def test_diagnostic_turn_callbacks_from_real_worker_keep_controls_and_logs(turn_env, marker_home, monkeypatch, setting, caplog):
    from tui_gateway import server_requests
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text(yaml.safe_dump(
        {} if setting is None else {"display": {"suppress_warning_notifications": setting}}))
    frames, controls, failures = [], [], []
    def wire(frame):
        frames.append(frame)
        if frame.get("method") == "clarify":
            assert server_requests.resolve_response({"id": frame["id"], "result": {"answer": "yes"}})
        return True
    monkeypatch.setattr(server, "write_json", wire)
    monkeypatch.setattr(server_requests, "_write", wire)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (server.threading.Event(), SimpleNamespace(join=lambda: None)))
    # The normal worker starts after the submitter releases _sessions_lock.
    # The shared inline-thread fixture instead holds that lock while joining
    # this real callback thread, creating a fixture-only deadlock.
    monkeypatch.setattr(server.threading, "Thread", server._RealThread)
    agent = Agent()
    agent.session_id = "session-key"
    agent.clear_interrupt = lambda: None
    session = _session(agent=agent, profile_home=str(marker_home), running=True)
    monkeypatch.setitem(server._sessions, "thread-owner", session)
    for key, callback in server._agent_cbs("thread-owner").items():
        setattr(agent, key, callback)
    def run(message, **kwargs):
        def worker():
            try:
                # Concrete callbacks can be retained by provider/helper workers.
                agent._emit_status("plain wake progress")
                agent._emit_wait_notice("plain wake thinking")
                agent._emit_notice(SimpleNamespace(level="info", text="wake notice", kind="custom", ttl_ms=0, key="custom", id="n"))
                kwargs["stream_callback"]("wake stream")
                controls.append(agent.clarify_callback("Continue recovery?", ["yes", "no"]))
                logging.getLogger("operator").warning("worker diagnostic retained")
            except BaseException as exc:
                failures.append(exc)
        thread = server._RealThread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        return {"final_response": "wake result", "messages": []}
    agent.run_conversation = run
    with caplog.at_level(logging.WARNING, logger="operator"):
        server._run_prompt_submit("request", "thread-owner", session, "engine failure",
            display_metadata={"notification_category": "diagnostic"})
        session["_run_thread"].join(timeout=10)
        assert not session["_run_thread"].is_alive()
    assert not failures
    assert controls == ["yes"]
    assert "worker diagnostic retained" in caplog.text
    presentation = [f for f in frames if f.get("params", {}).get("type") in
                    {"status.update", "thinking.delta", "notification.show", "message.delta", "message.complete"}]
    assert len(presentation) == (0 if setting is True else 5)
    # Once the turn completes, the same sink must present the next requested progress.
    agent._emit_status("next requested progress")
    assert frames[-1]["params"]["payload"]["text"] == "next requested progress"
