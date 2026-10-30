"""A leftover adviser steer becomes the next turn; its row must stay an adviser card."""

from __future__ import annotations

import threading

from tui_gateway import prompt_turn


def _run(monkeypatch, result):
    session = {"history_lock": threading.RLock(), "transport": None}
    rows: list[tuple[dict, str | None]] = []
    monkeypatch.setattr(prompt_turn, "_enqueue_prompt", lambda s, t, tr: {"text": t}, raising=False)
    monkeypatch.setattr(
        prompt_turn, "_persist_queued_user_row", lambda s, e, k: rows.append((e, k)),
        raising=False)
    monkeypatch.setattr(prompt_turn, "_drain_queued_prompt", lambda *a: True, raising=False)
    prompt_turn._run_post_turn_followups(None, "session-1", session, result, None)
    return rows


def test_leftover_adviser_steer_is_labelled(monkeypatch):
    rows = _run(monkeypatch, {
        "pending_steer": "[adviser — advisory only]\n- [concern · reviewer] check it",
        "pending_steer_kind": "adviser",
    })
    assert len(rows) == 1
    envelope, kind = rows[0]
    assert kind == "adviser"
    assert envelope["text"].startswith("[adviser")


def test_leftover_human_steer_stays_an_ordinary_turn(monkeypatch):
    rows = _run(monkeypatch, {"pending_steer": "typed by the user"})
    assert rows == [({"text": "typed by the user"}, None)]
