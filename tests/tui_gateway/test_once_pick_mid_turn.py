"""A ``/model X --once`` picked while a turn streams covers exactly the next turn.

``config.set model`` on a running session stashes the pick; ``_apply_pending_model_switch``
applies it at the next turn start, and a ``--once`` switch records its restore snapshot in
``session["one_turn_model_restore"]``. The turn thread had already taken that slot before
the switch wrote it, so the restore was picked up by the FOLLOWING turn: the once-model
answered two turns instead of one.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Agent:
    def __init__(self):
        self.model, self.provider, self.api_key, self.base_url, self.api_mode = (
            "base-model", "custom", "sk-fake", "http://127.0.0.1:9/v1", "chat_completions")
        self.session_id = "agent-sid"
        self.turn_models: list[str] = []

    def switch_model(self, *, new_model, new_provider, api_key, base_url, api_mode, capabilities=None):
        self.model, self.provider = new_model, new_provider

    def run_conversation(self, *args, **kwargs):
        self.turn_models.append(self.model)
        return {"final_response": "ok"}

    def clear_interrupt(self):
        pass


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    for name in ("_emit", "_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_tts_stream_begin", "_sync_session_key_after_compress", "_restart_slash_worker",
                 "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})

    def switch_model(*, raw_input, **_kwargs):
        return types.SimpleNamespace(
            success=True, new_model=raw_input, target_provider="custom", api_key="sk-fake",
            base_url="http://127.0.0.1:9/v1", api_mode="chat_completions", warning_message="",
            model_info=None, runtime_capabilities=None, error_message="")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch_model)


def test_once_pick_queued_mid_turn_answers_one_turn_only(turn_env):
    agent = _Agent()
    session = {
        "agent": agent, "session_key": "gw-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        # What config.set leaves behind for ``/model once-model --once`` sent while a turn streamed.
        "pending_model_switch": {"raw": "once-model --once", "confirm_expensive_model": True},
    }

    for text in ("first turn after the pick", "second turn"):
        session["running"] = True
        server._run_prompt_submit("rid", "ui-sid", session, text)

    assert agent.turn_models == ["once-model", "base-model"]
    assert "one_turn_model_restore" not in session
