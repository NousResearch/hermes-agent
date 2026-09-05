"""Live-owner Bot Mode DM handoff (#103030) — the gateway's registered deliverer.

The LOCAL ``message_agent`` path (``tools/bot_mode_dm``) used to spawn a second CLI for
the target's canonical Bot Chat; when THIS gateway hosts that chat live, the single-owner
lease fences the subprocess out and the DM is refused (``target_busy``) with no turn ever
run. The gateway registers ``_deliver_dm_to_live_session`` so the tool submits through
``prompt.submit(queued=True)`` instead — the same mechanism ``bot_relay.deliver`` uses for
relayed DMs (#101293). No live session for the profile → ``None`` → subprocess fallback.
"""

import pytest

from tools import bot_mode_dm
from tui_gateway import server


def test_gateway_registers_live_dm_deliverer():
    """Server import must wire the handoff into bot_mode_dm — without it the local path
    stays on the subprocess transport and live-owned targets keep getting refused."""
    assert server._deliver_dm_to_live_session in bot_mode_dm._LIVE_DM_DELIVERERS


def test_deliverer_submits_queued_into_live_bot_chat(monkeypatch):
    profile_home = "/srv/hermes/profiles/nezuko"
    record = {"profile_home": profile_home, "pending_title": "Bot Chat"}
    monkeypatch.setattr(server, "_sessions", {"live-sid": record})
    monkeypatch.setattr(server, "_profile_home", lambda profile: profile_home)
    submitted: list[tuple[int, dict]] = []

    def _fake_prompt_submit(rid, params):
        submitted.append((rid, params))
        return {"jsonrpc": "2.0", "id": rid, "result": {"ok": True}}

    monkeypatch.setattr(server, "_methods", {"prompt.submit": _fake_prompt_submit})

    receipt = server._deliver_dm_to_live_session("nezuko", "Message from 🤖 tanjiro (@tanjiro): ping")

    assert receipt is not None
    assert receipt["status"] == "sent"
    assert receipt["to"] == "@nezuko"
    assert "open Bot Chat" in receipt["detail"]
    assert [(rid, params) for rid, params in submitted] == [
        (0, {"session_id": "live-sid", "text": "Message from 🤖 tanjiro (@tanjiro): ping", "queued": True})
    ]


def test_deliverer_returns_none_without_live_session(monkeypatch):
    """No live Bot Chat for the profile → None → the subprocess transport runs as before."""
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_profile_home", lambda profile: None)
    submitted: list[tuple[int, dict]] = []

    def _fake_prompt_submit(rid, params):
        submitted.append((rid, params))
        return {"jsonrpc": "2.0", "id": rid, "result": {"ok": True}}

    monkeypatch.setattr(server, "_methods", {"prompt.submit": _fake_prompt_submit})

    assert server._deliver_dm_to_live_session("nezuko", "ping") is None
    assert submitted == []


def test_deliverer_defers_to_subprocess_transport_on_submit_error(monkeypatch):
    """A prompt.submit error surfaces through the subprocess transport's error shaping
    (target_busy refusal + retry policy) instead of a half-acknowledged delivery."""
    profile_home = "/srv/hermes/profiles/nezuko"
    record = {"profile_home": profile_home, "pending_title": "Bot Chat"}
    monkeypatch.setattr(server, "_sessions", {"live-sid": record})
    monkeypatch.setattr(server, "_profile_home", lambda profile: profile_home)

    def _failing_prompt_submit(rid, params):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": 4009, "message": "session busy"}}

    monkeypatch.setattr(server, "_methods", {"prompt.submit": _failing_prompt_submit})

    assert server._deliver_dm_to_live_session("nezuko", "ping") is None
