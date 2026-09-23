"""Turn-token stamps on the prompt-submit stream (#119543).

``message.start`` and ``message.complete`` for the SAME accepted prompt must
carry the same non-empty ``turn`` token, so a late complete from a superseded
turn can be dropped client-side instead of appending after the next user
message. Unstamped emitters (auto-continue bookkeeping, notifications) still
emit payload ``None`` — their completes stay inert-compatible.

Regression for #119543.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously so tests observe its emissions."""

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
def emits(monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _run_successful_turn(emits) -> tuple[dict, dict]:
    before = len(emits)
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._run_prompt_submit("rid", "ui-sid", session, "go")

    batch = emits[before:]
    starts = [payload for event, _sid, payload in batch if event == "message.start"]
    completes = [payload for event, _sid, payload in batch if event == "message.complete"]
    assert len(starts) == 1, f"expected one message.start, got {len(starts)}"
    assert len(completes) == 1, f"expected one message.complete, got {len(completes)}"
    return starts[0], completes[0]


def test_start_and_complete_share_one_turn_token(emits, turn_env):
    """The token minted at accept is stamped on BOTH frames of the same turn."""
    start, complete = _run_successful_turn(emits)

    token = start.get("turn")
    assert isinstance(token, str) and token
    assert complete.get("turn") == token


def test_turn_token_is_unique_per_accepted_prompt(emits, turn_env):
    """Two accepted prompts mint distinct tokens — never a shared identity."""
    first_start, first_complete = _run_successful_turn(emits)
    second_start, second_complete = _run_successful_turn(emits)

    assert first_start["turn"] != second_start["turn"]
    assert first_complete["turn"] != second_complete["turn"]


def test_unmuted_start_still_emits_with_payload_object(emits, turn_env):
    """Non-muted prompts keep emitting start with a payload (now the token)."""
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    server._run_prompt_submit("rid", "ui-sid", session, "go")

    starts = [(event, sid, payload) for event, sid, payload in emits if event == "message.start"]
    assert starts, "message.start must still be emitted for an unmuted turn"
    _event, sid, payload = starts[0]
    assert sid == "ui-sid"
    assert isinstance(payload, dict)
    assert isinstance(payload.get("turn"), str) and payload["turn"]
