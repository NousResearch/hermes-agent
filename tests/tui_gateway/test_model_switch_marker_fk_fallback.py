"""Regression test: FK fallback in _append_model_switch_marker.

When the primary persist (targeted at ``_submit_row_target_key``) fails with an
FK violation — because the rotated ``agent.session_id`` row was never created in
the DB — the marker must fall back to ``session_key``, which
``_ensure_session_db_row`` is guaranteed to have created.

Without the fallback the model-switch notice is silently lost: the next resume
replays without it and nothing lands in errors.log.
"""

from __future__ import annotations

import contextlib
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway.server import _append_model_switch_marker


class _FakeDB:
    """SessionDB stub that FK-fails on the rotated target but succeeds on session_key."""

    def __init__(self, *, fail_on_target: str) -> None:
        self.fail_on_target = fail_on_target
        self.appended: list[tuple[str, dict]] = []
        self.deactivated: list[str] = []

    def deactivate_messages_by_display_kind(self, session_id: str, kind: str) -> None:
        self.deactivated.append(session_id)

    def append_message(self, *, session_id: str, role: str, content: str,
                       display_kind: str | None = None, message_uid: str | None = None) -> int:
        if session_id == self.fail_on_target:
            raise Exception("FOREIGN KEY constraint failed")
        self.appended.append((session_id, {
            "role": role, "content": content,
            "display_kind": display_kind, "message_uid": message_uid,
        }))
        return 42


class TestModelSwitchMarkerFKFallback:
    """The FK fallback must persist the marker under session_key when the
    rotated target row does not exist."""

    def _make_session(self, *, session_key: str, agent_session_id: str) -> dict:
        agent = types.SimpleNamespace(session_id=agent_session_id, _session_db=None)
        return {
            "agent": agent,
            "session_key": session_key,
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
        }

    def test_fallback_persists_under_session_key(self, monkeypatch) -> None:
        """Primary write FK-fails on rotated target; fallback lands on session_key."""
        session = self._make_session(
            session_key="parent-key",
            agent_session_id="rotated-child-id",
        )
        fake_db = _FakeDB(fail_on_target="rotated-child-id")

        # _ensure_session_db_row is a no-op (row already exists under session_key).
        monkeypatch.setattr(
            "tui_gateway.server._ensure_session_db_row", lambda s: True
        )
        # _session_db yields our fake.
        def _fake_session_db(s):
            @contextlib.contextmanager
            def _cm():
                yield fake_db
            return _cm()
        monkeypatch.setattr("tui_gateway.server._session_db", _fake_session_db)

        _append_model_switch_marker(session, model="gpt-4o", provider="openai")

        # The fallback must have appended under session_key, NOT the rotated target.
        assert len(fake_db.appended) == 1, (
            f"Expected exactly 1 append (the fallback), got {len(fake_db.appended)}"
        )
        appended_key, appended_msg = fake_db.appended[0]
        assert appended_key == "parent-key", (
            f"Expected fallback to session_key 'parent-key', got '{appended_key}'"
        )
        assert appended_msg["display_kind"] == "model_switch"
        assert "gpt-4o" in appended_msg["content"]

        # The entry must be stamped as persisted.
        from agent.context_compressor import _DB_PERSISTED_MARKER
        entry = session["history"][-1]
        assert entry.get(_DB_PERSISTED_MARKER) is True
        assert entry.get("_row_id") == 42

    def test_no_fallback_when_target_equals_session_key(self, monkeypatch) -> None:
        """When target == session_key (no rotation), a failure must NOT retry
        (the fallback guard requires session_key != target)."""
        session = self._make_session(
            session_key="same-key",
            agent_session_id="same-key",
        )
        fake_db = _FakeDB(fail_on_target="same-key")

        monkeypatch.setattr(
            "tui_gateway.server._ensure_session_db_row", lambda s: True
        )
        def _fake_session_db(s):
            @contextlib.contextmanager
            def _cm():
                yield fake_db
            return _cm()
        monkeypatch.setattr("tui_gateway.server._session_db", _fake_session_db)

        _append_model_switch_marker(session, model="gpt-4o", provider="openai")

        # No fallback: the single attempt failed and nothing was appended.
        assert len(fake_db.appended) == 0, (
            "Fallback must NOT fire when target == session_key"
        )

    def test_no_fallback_when_target_empty(self, monkeypatch) -> None:
        """If _submit_row_target_key never ran (exception before assignment),
        target stays '' and the fallback guard (target truthy) blocks the retry."""
        session = self._make_session(
            session_key="parent-key",
            agent_session_id="rotated-child-id",
        )
        # Make _submit_row_target_key raise so target stays "".
        monkeypatch.setattr(
            "tui_gateway.server._submit_row_target_key",
            lambda s: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        fake_db = _FakeDB(fail_on_target="rotated-child-id")
        monkeypatch.setattr(
            "tui_gateway.server._ensure_session_db_row", lambda s: True
        )
        def _fake_session_db(s):
            @contextlib.contextmanager
            def _cm():
                yield fake_db
            return _cm()
        monkeypatch.setattr("tui_gateway.server._session_db", _fake_session_db)

        _append_model_switch_marker(session, model="gpt-4o", provider="openai")

        assert len(fake_db.appended) == 0, (
            "Fallback must NOT fire when target is empty (never resolved)"
        )

    def test_fallback_still_fails_gracefully(self, monkeypatch) -> None:
        """If the fallback write ALSO fails, the exception is swallowed (warning
        logged) and the function returns without raising."""
        session = self._make_session(
            session_key="parent-key",
            agent_session_id="rotated-child-id",
        )
        # DB that fails on EVERY append.
        class _AllFailDB:
            def deactivate_messages_by_display_kind(self, sid, kind):
                pass
            def append_message(self, **kw):
                raise Exception("disk full")

        fake_db = _AllFailDB()
        monkeypatch.setattr(
            "tui_gateway.server._ensure_session_db_row", lambda s: True
        )
        def _fake_session_db(s):
            @contextlib.contextmanager
            def _cm():
                yield fake_db
            return _cm()
        monkeypatch.setattr("tui_gateway.server._session_db", _fake_session_db)

        # Must not raise.
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")

        # The in-memory marker is still present (history is updated before the DB write).
        assert len(session["history"]) == 1
        assert "gpt-4o" in session["history"][0]["content"]
