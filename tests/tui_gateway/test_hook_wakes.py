"""An ``async_rewake`` shell hook's wake (``hook_wake``) on the TUI / Desktop poller: only the session
whose key it carries runs it, as a turn typed ``internal_notification`` with a one-line title (never a
user bubble of the hook's text), and ``display.background_process_notifications: off`` does not mute
it: the hook is an explicit opt-in, not background-process chatter.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from tools.process_registry_notifications import format_process_notification
from tui_gateway import server

HOOK_WAKE = {"type": "hook_wake", "event_id": "e1", "session_id": "hook_e1", "session_key": "chat-a",
             "command": "agent-bus ready", "hook_event": "post_llm_call", "output": "ping from lex", "elapsed": 60}


@pytest.fixture
def surface(monkeypatch, tmp_path):
    submits: list = []
    statuses: list = []
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, session, text, **kw: submits.append((text, kw)) or True)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: statuses.append(payload))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"background_process_notifications": "off"}}), encoding="utf-8")
    return submits, statuses


def _session(tmp_path, key: str) -> dict:
    return {"history_lock": threading.RLock(), "running": False, "history": [], "agent": None,
            "profile_home": str(tmp_path), "session_key": key}


def _handle(session: dict, sid: str) -> bool:
    registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda session_id: False)
    return server._notif_handle_event(sid, session, dict(HOOK_WAKE), set(), registry, format_process_notification, [])


def test_the_owning_session_runs_it_typed_and_titled_even_with_process_notifications_off(surface, tmp_path):
    submits, statuses = surface
    assert _handle(_session(tmp_path, "chat-a"), "sid-a") is True
    ((text, kwargs),) = submits
    assert text == format_process_notification(HOOK_WAKE)
    assert (kwargs["display_kind"], kwargs["display_metadata"]) == (
        "internal_notification", {"display_text": "Shell Hook Wake: agent-bus ready"})
    assert [s["text"] for s in statuses] == ["Shell Hook Wake: agent-bus ready"]


def test_another_session_never_runs_it(surface, tmp_path):
    submits, _statuses = surface
    _handle(_session(tmp_path, "chat-b"), "sid-b")
    assert submits == []
