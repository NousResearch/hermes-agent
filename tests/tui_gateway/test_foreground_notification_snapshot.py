"""Foreground decisions stay stable through file changes and child callbacks."""
import threading
from types import SimpleNamespace

import pytest
import yaml
from agent.status_output import StatusOutputMixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from tui_gateway import server
from tests.tui_gateway.test_auto_continue import turn_env, marker_home, _session


class Agent(StatusOutputMixin):
    suppress_status_output = True
    platform = "tui"
    session_id = "session-key"
    def clear_interrupt(self):
        pass


class CLI(CLIStreamMixin):
    def _invalidate(self):
        pass


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_tui_real_turn_snapshot_next_turn_and_child_threads(turn_env, marker_home, monkeypatch, initial, fail):
    home = marker_home / "snapshot-owner"
    home.mkdir()
    cfg = home / "config.yaml"
    def configure(value):
        cfg.write_text(yaml.safe_dump({"display": {"suppress_warning_notifications": value}}))
    configure(initial)
    frames, seen, errors = [], [], []
    monkeypatch.setattr(server, "write_json", lambda f: frames.append(f) or True)
    monkeypatch.setattr(server.threading, "Thread", server._RealThread)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    agent = Agent()
    session = _session(agent=agent, profile_home=str(home))
    monkeypatch.setitem(server._sessions, "snapshot", session)
    for name, cb in server._agent_cbs("snapshot").items():
        setattr(agent, name, cb)
    expected = [initial, not initial]
    def run(message, **kwargs):
        snapshot = expected[len(seen)]
        configure(not snapshot)
        before = len(frames)
        def child():
            try:
                agent._emit_warning("same foreground warning")
            except BaseException as exc:
                errors.append(exc)
        worker = server._RealThread(target=child)
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        warnings = [f for f in frames[before:] if f.get("params", {}).get("type") == "status.update"]
        seen.append(len(warnings))
        if fail:
            raise RuntimeError("fixture turn failure")
        return {"final_response": "requested result", "messages": []}
    agent.run_conversation = run
    for _ in expected:
        assert server._run_prompt_submit("r", "snapshot", session, "human request")
        session["_run_thread"].join(10)
        assert not session["_run_thread"].is_alive()
        assert not hasattr(agent, "_notification_config")
    assert not errors
    assert seen == [0 if value else 1 for value in expected]


@pytest.mark.parametrize("initial", [False, True])
def test_cli_callbacks_honor_bound_snapshot_not_file(tmp_path, monkeypatch, initial):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = Agent()
    agent._notification_config = {"display": {"suppress_warning_notifications": initial}}
    surface = CLI()
    surface.agent = agent
    for changed in (not initial, initial, not initial):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": {"suppress_warning_notifications": changed}}))
        surface._pending_credit_notices = []
        surface._on_notice(SimpleNamespace(level="warn", text="same warning"))
        assert bool(surface._pending_credit_notices) is not initial


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_real_chat_binds_refreshes_and_restores_snapshot(tmp_path, monkeypatch, initial, fail):
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = tmp_path / "config.yaml"
    def configure(value):
        cfg.write_text(yaml.safe_dump({"display": {"suppress_warning_notifications": value}}))
    configure(initial)
    seen = []
    class Surface(CLIChatTurnMixin, CLI):
        _active_agent_route_signature = "fixed"
        _secret_capture_callback = None
        def _ensure_runtime_credentials(self): return True
        def _resolve_turn_agent_config(self, message): return {"signature": "fixed", "model": "fixture", "runtime": {}}
        def _init_agent(self, **kwargs): return True
        def _chat_route_images(self, message, images): return message
        def _chat_expand_context_references(self, message): return message, None
        def _chat_stage_user_message(self, agent, message): agent._pending_cli_user_message = {}
        def _reset_stream_state(self): pass
        def _chat_setup_turn_audio(self, *args): pass
        def _chat_release_turn_audio(self, *args): pass
        def _chat_settle_turn(self, turn):
            if fail: raise RuntimeError("fixture settle error")
        def _chat_render_turn(self, *args): return "requested result"
        def _chat_monitor_agent_thread(self, turn, worker):
            worker.join(5)
            assert not worker.is_alive()
        def _chat_run_agent(self, turn, message):
            expected = [initial, not initial][len(seen)]
            configure(not expected)
            self._pending_credit_notices = []
            self._on_notice(SimpleNamespace(level="warn", text="worker warning"))
            seen.append(bool(self._pending_credit_notices))
    surface = Surface()
    surface.agent = Agent()
    for _ in range(2):
        result = surface.chat("human request")
        assert result == (None if fail else "requested result")
        assert not hasattr(surface.agent, "_notification_config")
        assert not hasattr(surface.agent, "_notification_platform")
    assert seen == [not initial, initial]
