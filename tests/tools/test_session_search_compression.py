"""Tests for session_search compression-continuation visibility (#13840).

Compression-ended sessions have parent_session_id set (they are continuation
children of compressed parents).  The blanket parent_session_id filter
excluded them from search results, making compressed sessions invisible.
"""
import json
import pytest
from unittest.mock import MagicMock, patch


class TestListRecentSessionsCompression:
    """Compression-continuation sessions must be visible in recent sessions (#13840)."""

    def _run_list_recent(self, sessions, current_session_id=None):
        from tools.session_search_tool import _list_recent_sessions
        db = MagicMock()
        db.list_sessions_rich.return_value = sessions
        db.get_session.return_value = None
        return json.loads(_list_recent_sessions(db, limit=10, current_session_id=current_session_id))

    def test_compression_continuation_visible(self):
        """Sessions with parent_session_id AND end_reason=compression must be visible."""
        sessions = [
            {"id": "session-2", "parent_session_id": "session-1", "end_reason": "compression",
             "title": "Continued after compression", "source": "cli", "started_at": "", "last_active": "",
             "message_count": 5, "preview": "hello"},
        ]
        result = self._run_list_recent(sessions)
        assert result["success"]
        assert len(result["results"]) == 1
        assert result["results"][0]["session_id"] == "session-2"

    def test_delegation_child_still_hidden(self):
        """Sessions with parent_session_id but no compression end_reason stay hidden."""
        sessions = [
            {"id": "delegate-1", "parent_session_id": "parent-1", "end_reason": None,
             "title": "Delegated task", "source": "cli", "started_at": "", "last_active": "",
             "message_count": 3, "preview": "sub-task"},
        ]
        result = self._run_list_recent(sessions)
        assert len(result["results"]) == 0

    def test_root_sessions_still_visible(self):
        """Sessions without parent_session_id are always visible."""
        sessions = [
            {"id": "root-1", "parent_session_id": None, "title": "Root session",
             "source": "cli", "started_at": "", "last_active": "", "message_count": 10, "preview": "hi"},
        ]
        result = self._run_list_recent(sessions)
        assert len(result["results"]) == 1
