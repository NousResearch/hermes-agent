"""The TUI's rewind helper refuses a target that is no longer in the session's history with ``ValueError``, and the
stored transcript stays as it was."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import conversation, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_a_vanished_target_raises_value_error_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tui_gateway.server import _rewind_active_session_history

    every_row = TranscriptRead(view=TranscriptView.ALL)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "oob-tui", "cli")
        seed_messages(store, "oob-tui", {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"})
        before = transcript_rows(store, "oob-tui", every_row)
        warm = conversation(store, "oob-tui")
        session = {"agent": SimpleNamespace(_session_messages=warm), "history": list(warm),
                   "history_lock": threading.Lock(), "history_version": 0, "session_key": "oob-tui"}
        with pytest.raises(ValueError):  # the TUI's own error class for a vanished target
            _rewind_active_session_history(session, -1)  # /undo 2 on a one-turn history
        assert transcript_rows(store, "oob-tui", every_row) == before
