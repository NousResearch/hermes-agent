"""Regression coverage for the warm/live session display projection.

``_live_visible_history`` reads the persisted display lineage — and, since
#92080, it must pass ``include_compacted=True`` so a compacted session's
archived turns survive a warm session switch (without it the chat repainted
as just summary + tail while REST showed everything). These tests pin the
``include_compacted`` flag and the reconcile/fallback contract.
"""

from tui_gateway.server import _live_visible_history


class _FakeDB:
    def __init__(self, rows=None, error=None):
        self.calls = []
        self._rows = rows if rows is not None else []
        self._error = error

    def get_messages_as_conversation(self, key, **kwargs):
        self.calls.append((key, kwargs))
        if self._error is not None:
            raise self._error
        return self._rows


def _msg(role, text):
    return {"role": role, "content": text}


class TestIncludeCompactedFlag:
    def test_compacted_flag_is_passed_to_the_db_read(self):
        """#92080: the display read must include compacted (archived) turns."""
        db = _FakeDB(rows=[_msg("user", "hi")])
        _live_visible_history({"session_key": "s1"}, db, [])
        assert len(db.calls) == 1
        key, kwargs = db.calls[0]
        assert key == "s1"
        assert kwargs.get("include_compacted") is True
        assert kwargs.get("include_ancestors") is True
        assert kwargs.get("include_row_ids") is True


class TestReconcileContract:
    def test_db_display_used_when_memory_empty(self):
        rows = [_msg("user", "hi"), _msg("assistant", "hello")]
        db = _FakeDB(rows=rows)
        result = _live_visible_history({"session_key": "s1"}, db, [])
        assert result == rows

    def test_unflushed_memory_tail_appended_after_db_anchor(self):
        db_rows = [_msg("user", "hi"), _msg("assistant", "hello")]
        db = _FakeDB(rows=db_rows)
        memory = [
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "not flushed yet"),
        ]
        result = _live_visible_history({"session_key": "s1"}, db, memory)
        assert result == db_rows + [_msg("user", "not flushed yet")]


class TestFallbacks:
    def test_db_read_failure_falls_back_to_memory(self):
        db = _FakeDB(error=RuntimeError("db gone"))
        memory = [_msg("user", "in memory only")]
        result = _live_visible_history({"session_key": "s1"}, db, memory)
        assert result == memory

    def test_no_db_falls_back_to_memory(self):
        memory = [_msg("user", "in memory only")]
        result = _live_visible_history({"session_key": "s1"}, None, memory)
        assert result == memory

    def test_no_session_key_falls_back_to_memory(self):
        memory = [_msg("user", "in memory only")]
        db = _FakeDB(rows=[_msg("user", "ignored")])
        result = _live_visible_history({}, db, memory)
        assert result == memory
        assert db.calls == []
