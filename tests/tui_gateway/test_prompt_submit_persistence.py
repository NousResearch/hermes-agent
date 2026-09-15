import contextlib
import threading

from agent.context_compressor import _DB_PERSISTED_MARKER
from tui_gateway import server


class _RecordingDb:
    def __init__(self):
        self.calls = []

    def append_messages_batch(self, session_id, messages):
        self.calls.append((session_id, messages))


def test_prompt_submit_persists_user_before_deferred_agent_build(monkeypatch):
    """Regression for #111868: an accepted first send survives process loss while the agent is still building."""
    db = _RecordingDb()
    session = {
        "attached_images": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-first-turn",
    }
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_session_db", lambda _session: contextlib.nullcontext(db))

    assert server._persist_session_row_for_submit(1, session, "keep this message") is None
    assert db.calls == [
        (
            "stored-first-turn",
            [
                {
                    "role": "user",
                    "content": "keep this message",
                    "timestamp": session["_prepersisted_user_message"]["timestamp"],
                }
            ],
        )
    ]
    assert session["_prepersisted_user_message"][_DB_PERSISTED_MARKER] is True
