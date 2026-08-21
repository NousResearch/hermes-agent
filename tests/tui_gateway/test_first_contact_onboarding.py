"""Desktop/TUI first-contact profile-build onboarding via tui_gateway."""

from __future__ import annotations

import threading
import types

import pytest
import yaml

from agent.onboarding import PROFILE_BUILD_FLAG, profile_build_directive
from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def marker_home(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def turn_env(monkeypatch, tmp_path, marker_home):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_pending_reaction_notes", lambda _s: "")


def test_run_prompt_submit_stages_profile_build_on_first_contact(
    monkeypatch, tmp_path, emits, turn_env, marker_home
):
    """Fresh install + empty history must stage the opt-in profile-build note."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"onboarding": {"profile_build": "ask"}}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    staged: list[str] = []

    def _run(message, **kwargs):
        staged.append(getattr(agent, "_gateway_turn_context_notes", ""))
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "sid", session, "hello there")

    assert staged == [profile_build_directive().strip()]
    loaded = yaml.safe_load(cfg_path.read_text())
    assert loaded["onboarding"]["seen"][PROFILE_BUILD_FLAG] is True


def test_run_prompt_submit_skips_first_contact_when_prior_sessions_exist(
    emits, turn_env, marker_home, monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: True)

    staged: list[str] = []

    def _run(message, **kwargs):
        staged.append(getattr(agent, "_gateway_turn_context_notes", ""))
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(agent=agent, running=True)

    server._run_prompt_submit("rid", "sid", session, "hello again")

    assert staged == [""]


def test_run_prompt_submit_skips_first_contact_when_history_not_empty(
    emits, turn_env, marker_home, monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    staged: list[str] = []

    def _run(message, **kwargs):
        staged.append(getattr(agent, "_gateway_turn_context_notes", ""))
        return {"final_response": "done"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=_run, clear_interrupt=lambda: None
    )
    session = _session(
        agent=agent,
        running=True,
        history=[{"role": "user", "content": "prior"}],
    )

    server._run_prompt_submit("rid", "sid", session, "follow up")

    assert staged == [""]
