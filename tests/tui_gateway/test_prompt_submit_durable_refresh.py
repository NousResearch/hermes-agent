from __future__ import annotations

import contextlib
import logging
import threading

from tui_gateway.prompt_history_sync import (
    refresh_resumed_history_before_submit,
    wrap_prompt_submit,
)


class _FakeDB:
    def __init__(self, history, *, on_read=None):
        self.history = list(history)
        self.on_read = on_read
        self.reads = 0

    def get_resume_conversations(self, session_key):
        assert session_key == "cron_session"
        self.reads += 1
        if self.on_read is not None:
            self.on_read()
        return list(self.history), list(self.history)


class _FakeServer:
    def __init__(self, session, db):
        self._sessions = {"live-sid": session}
        self._db = db
        self.logger = logging.getLogger("test_prompt_submit_durable_refresh")

    def _session_db(self, _session):
        return contextlib.nullcontext(self._db)

    @staticmethod
    def sanitize_replay_history(history):
        return list(history)


def _messages(count):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(count)
    ]


def _resumed_session(history):
    ready = threading.Event()
    ready.set()
    return {
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "resume_history_ready": ready,
        "resume_session_id": "cron_session",
        "running": False,
        "session_key": "cron_session",
    }


def test_wrap_refreshes_open_time_prefix_before_prompt_handler_runs():
    """Regression for #91508: the handler must see all externally persisted rows."""
    opened_at = _messages(12)
    durable_after_cron_finished = _messages(37)
    session = _resumed_session(opened_at)
    db = _FakeDB(durable_after_cron_finished)
    server = _FakeServer(session, db)
    seen = {}

    def handler(_rid, _params):
        seen["history"] = list(session["history"])
        return {"ok": True}

    result = wrap_prompt_submit(server, handler)(
        "1", {"session_id": "live-sid", "text": "follow-up"}
    )

    assert result == {"ok": True}
    assert len(seen["history"]) == 37
    assert seen["history"][-1]["content"] == "m36"
    assert session["history_version"] == 1
    assert db.reads == 1


def test_refresh_does_not_touch_new_non_resumed_sessions():
    session = _resumed_session(_messages(2))
    session.pop("resume_session_id")
    db = _FakeDB(_messages(8))
    server = _FakeServer(session, db)

    assert refresh_resumed_history_before_submit(
        server, {"session_id": "live-sid", "text": "hello"}
    ) is False
    assert len(session["history"]) == 2
    assert session["history_version"] == 0
    assert db.reads == 0


def test_refresh_never_replaces_memory_with_a_shorter_durable_projection():
    session = _resumed_session(_messages(8))
    db = _FakeDB(_messages(6))
    server = _FakeServer(session, db)

    assert refresh_resumed_history_before_submit(
        server, {"session_id": "live-sid", "text": "hello"}
    ) is False
    assert len(session["history"]) == 8
    assert session["history_version"] == 0


def test_local_history_mutation_wins_if_it_races_the_durable_read():
    session = _resumed_session(_messages(12))

    def mutate_locally():
        with session["history_lock"]:
            session["history"] = _messages(13)
            session["history_version"] += 1

    db = _FakeDB(_messages(37), on_read=mutate_locally)
    server = _FakeServer(session, db)

    assert refresh_resumed_history_before_submit(
        server, {"session_id": "live-sid", "text": "hello"}
    ) is False
    assert len(session["history"]) == 13
    assert session["history_version"] == 1


def test_busy_session_keeps_the_active_turn_as_history_owner():
    session = _resumed_session(_messages(12))
    session["running"] = True
    db = _FakeDB(_messages(37))
    server = _FakeServer(session, db)

    assert refresh_resumed_history_before_submit(
        server, {"session_id": "live-sid", "text": "queued follow-up"}
    ) is False
    assert len(session["history"]) == 12
    assert db.reads == 0
