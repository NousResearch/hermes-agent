"""RED-first tests for exact-session TUI/dashboard injection (H-104, H-106).

Pins the host-owned seam ``tui_gateway.server.inject_external_message``:

- an idle dashboard session receives only its own targeted message;
- a busy dashboard session queues at the safe boundary without interrupting
  the active turn;
- unknown, missing or invalid targets fail closed (False);
- invalid modes are rejected.
"""

from __future__ import annotations

import threading

import pytest

import tui_gateway.server as server


class FakeAgent:
    def __init__(self) -> None:
        self.steered: list[str] = []

    def steer(self, payload: str) -> bool:
        self.steered.append(payload)
        return True


def _session(running: bool = False, agent: FakeAgent | None = None) -> dict:
    return {
        "agent": agent or FakeAgent(),
        "session_key": "tui:key",
        "running": running,
        "transport": object(),
        "queued_prompt": None,
        "_finalized": False,
        "history_lock": threading.Lock(),
    }


@pytest.fixture
def fake_sessions(monkeypatch):
    sessions: dict[str, dict] = {}
    submitted: list[tuple[str, str]] = []
    queued: list[tuple[str, str]] = []

    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_sessions_lock", threading.RLock())

    def fake_run_prompt_submit(rid, sid, session, text, **_kw):
        submitted.append((sid, str(text)))

    def fake_enqueue_prompt(session, text, transport, **_kw):
        queued.append((str(session["session_key"]), str(text)))

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_enqueue_prompt", fake_enqueue_prompt)

    return {
        "sessions": sessions,
        "submitted": submitted,
        "queued": queued,
    }


# ---------------------------------------------------------------------------
# H-104 — exact TUI target: idle and busy sessions receive only their own
# ---------------------------------------------------------------------------


class TestTuiExactTarget:
    def test_idle_session_receives_only_own_message(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-1"] = _session(running=False)
        sessions["sid-2"] = _session(running=False)

        ok = server.inject_external_message("for sid-1 only", target_session="sid-1")
        assert ok is True
        assert fake_sessions["submitted"] == [("sid-1", "for sid-1 only")]
        assert fake_sessions["queued"] == []

    def test_busy_session_queues_at_safe_boundary(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-1"] = _session(running=True)

        ok = server.inject_external_message("queued work", target_session="sid-1")
        assert ok is True
        assert fake_sessions["submitted"] == []
        assert fake_sessions["queued"] == [("tui:key", "queued work")]

    def test_two_busy_sessions_no_cross_session_leak(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-a"] = _session(running=True)
        sessions["sid-b"] = _session(running=True)

        assert server.inject_external_message("to a", target_session="sid-a") is True
        assert server.inject_external_message("to b", target_session="sid-b") is True
        keys = [k for k, _ in fake_sessions["queued"]]
        assert keys == ["tui:key", "tui:key"]  # each session's own transport key

    def test_target_by_session_key_resolves(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-9"] = _session(running=False)
        sessions["sid-9"]["session_key"] = "telegram:chat:42:user:7"

        ok = server.inject_external_message(
            "keyed target", target_session="telegram:chat:42:user:7"
        )
        assert ok is True
        assert fake_sessions["submitted"] == [("sid-9", "keyed target")]


# ---------------------------------------------------------------------------
# H-106 — closed/unknown targets fail closed
# ---------------------------------------------------------------------------


class TestTuiClosedTarget:
    def test_unknown_session_fails_closed(self, fake_sessions):
        assert server.inject_external_message("hi", target_session="nope") is False
        assert fake_sessions["submitted"] == []
        assert fake_sessions["queued"] == []

    def test_missing_target_rejected(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-1"] = _session(running=False)
        assert server.inject_external_message("hi") is False
        assert fake_sessions["submitted"] == []

    def test_finalized_session_rejected(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-1"] = _session(running=False)
        sessions["sid-1"]["_finalized"] = True
        assert server.inject_external_message("hi", target_session="sid-1") is False
        assert fake_sessions["submitted"] == []

    def test_invalid_mode_rejected(self, fake_sessions):
        sessions = fake_sessions["sessions"]
        sessions["sid-1"] = _session(running=False)
        assert server.inject_external_message("hi", target_session="sid-1", mode="bogus") is False
        assert fake_sessions["submitted"] == []
