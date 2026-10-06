"""Desktop/TUI turn-dispatch observability (#86647).

During the #79278/#86647 persistent-mute investigation the decisive evidence
was an *absence*: a Desktop request left no INFO record in ``agent.log`` or
``gateway.log`` at all (``0 platform=desktop`` across the whole file), so a
muted window was structurally indistinguishable from a request that never
arrived. This suite pins the two-record contract that fixes that:

* ``_run_prompt_submit`` logs one ``tui prompt accepted`` INFO record before
  the turn thread starts, carrying the UI session id, the gateway
  ``session_key``, and the agent's live ``session_id`` (rotated independently
  by compression — the triple is what a rotation-mute trace needs).
* The turn's ``finally`` logs exactly one ``tui turn finished`` bookend on
  every path (success, returned error, exception), re-reading
  ``agent.session_id`` so a mid-turn compression rotation shows up as an
  accepted/finished pair with different agent ids.
* No prompt content is ever logged.
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server

class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
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
        "session_key": "gw-session-key",
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
def turn_stubs(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})

@pytest.fixture()
def turn_env(turn_stubs, monkeypatch):
    """``turn_stubs`` with the turn run inline on the caller's thread."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)

def _records(caplog, needle):
    return [r for r in caplog.records if needle in r.getMessage()]

SECRETISH_PROMPT = "please rotate QDRANT_API_KEY=hunter2-super-secret now"

def test_accepted_and_finished_records_on_success(turn_env, caplog):
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, SECRETISH_PROMPT)

    accepted = _records(caplog, "tui prompt accepted")
    finished = _records(caplog, "tui turn finished")
    assert len(accepted) == 1
    assert len(finished) == 1

    msg = accepted[0].getMessage()
    # Prompt content is never logged — only its length.
    assert "hunter2" not in msg
    assert "QDRANT_API_KEY" not in msg

    fin = finished[0].getMessage()
    assert "hunter2" not in fin


@pytest.mark.parametrize("settle_info_raises", [False, True])
def test_turn_settles_before_post_turn_trim(turn_stubs, monkeypatch, caplog, settle_info_raises):
    """A blocked post-turn trim must not hold the session running or its bookend (#131740);
    turn audio still ends BEFORE settlement, so a next turn admitted during the trim keeps its own."""
    import hermes_cli.mem_trim as mem_trim
    import tools.voice_mode as voice_mode

    entered, release = threading.Event(), threading.Event()

    def blocking_trim(**_kw):
        entered.set()
        release.wait(timeout=10)

    monkeypatch.setattr(mem_trim, "trim_memory", blocking_trim)
    if settle_info_raises:  # a raising settle step must not skip the post-turn trim
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *a: 1 / 0)
    audio_end = []  # (event, session running at that moment)
    tts = types.SimpleNamespace(put=lambda x: x is None and audio_end.append(("tts", session["running"])))
    monkeypatch.setattr(server, "_start_turn_voice", lambda: (tts, True))
    monkeypatch.setattr(voice_mode, "stop_thinking_sound", lambda: audio_end.append(("thinking", session["running"])))
    agent = types.SimpleNamespace(
        session_id="agent-sid-1", run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None)
    session = _session(agent=agent, running=True)
    monkeypatch.setattr(server, "_sessions", {"ui-sid": session})
    try:
        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            assert server._run_prompt_submit("rid", "ui-sid", session, "hi")
            assert entered.wait(timeout=5)
            assert session["running"] is False
            assert len(_records(caplog, "tui turn finished")) == 1
            assert audio_end == [("thinking", True), ("tts", True)]
    finally:
        release.set()
        session["_run_thread"].join(timeout=5)
