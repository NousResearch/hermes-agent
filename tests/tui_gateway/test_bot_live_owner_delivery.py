"""TUI/Desktop live-owner intake for durable Bot Chat deliveries."""

from __future__ import annotations

import threading

from tui_gateway import server


def _session(*, running=False):
    return {
        "session_key": "canonical",
        "profile_home": "C:/profiles/target",
        "history_lock": threading.Lock(),
        "running": running,
        "transport": None,
    }


def test_idle_owner_claims_delivery_and_commits_terminal_receipt(monkeypatch):
    session = _session()
    claimed = {
        "id": "a" * 32,
        "session_id": "canonical",
        "message": "Message from agent: hello",
    }
    completions = []
    monkeypatch.setattr(
        "tools.bot_live_delivery.claim_pending_delivery",
        lambda home, session_id: claimed,
    )
    monkeypatch.setattr(
        "tools.bot_live_delivery.complete_delivery",
        lambda home, delivery_id, **payload: completions.append(
            (str(home), delivery_id, payload)
        ),
    )

    def run_prompt(rid, sid, live, text, **kwargs):
        assert text == claimed["message"]
        assert live["running"] is True
        kwargs["terminal_callback"]({"status": "settled", "text": "received"})
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    assert server._poll_bot_live_delivery_once("live-sid", session) is True
    assert completions == [
        (
            "C:/profiles/target",
            "a" * 32,
            {"status": "settled", "reply": "received", "error": "", "reason": ""},
        )
    ]


def test_busy_owner_leaves_request_unclaimed_for_bounded_sender_timeout(monkeypatch):
    session = _session(running=True)
    claims = []
    monkeypatch.setattr(
        "tools.bot_live_delivery.claim_pending_delivery",
        lambda *args: claims.append(args),
    )

    assert server._poll_bot_live_delivery_once("live-sid", session) is False
    assert claims == []


def test_receipt_write_failure_does_not_fail_dm_turn(monkeypatch):
    session = _session()
    claimed = {
        "id": "a" * 32,
        "session_id": "canonical",
        "message": "Message from agent: hello",
    }
    monkeypatch.setattr(
        "tools.bot_live_delivery.claim_pending_delivery", lambda *args: claimed
    )
    monkeypatch.setattr(
        "tools.bot_live_delivery.complete_delivery",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    completed = []

    def run_prompt(rid, sid, live, text, **kwargs):
        kwargs["terminal_callback"]({"status": "settled", "text": "received"})
        completed.append(text)
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    assert server._poll_bot_live_delivery_once("live-sid", session) is True
    assert completed == [claimed["message"]]
