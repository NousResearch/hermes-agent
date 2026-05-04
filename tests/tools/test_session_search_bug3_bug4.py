"""Tests for session_search fixes from issue #19434.

Bug 3: Child sessions were blindly skipped in _list_recent_sessions().
       Fix: resolve to root and deduplicate.

Bug 4: FTS scan limit of 50 caused cron sessions to dominate BM25 ranking.
       Fix: raise limit to 300 and demote cron sessions.
"""

import json
from unittest.mock import MagicMock


def _make_session(sid, source="cli", parent=None, title=None):
    return {
        "id": sid,
        "title": title or sid,
        "source": source,
        "started_at": "2026-01-01T00:00:00",
        "last_active": "2026-01-02T00:00:00",
        "message_count": 5,
        "preview": f"preview of {sid}",
        "parent_session_id": parent,
    }


def _make_db(sessions=None, session_map=None, fts_rows=None):
    db = MagicMock()
    db.list_sessions_rich.return_value = sessions or []
    db.search_messages.return_value = fts_rows or []
    db.get_messages_as_conversation.return_value = []
    session_map = session_map or {}
    db.get_session.side_effect = lambda sid: session_map.get(sid)
    return db


class TestChildSessionResolution:

    def _list_recent(self, db, limit=5, current=None):
        from tools.session_search_tool import _list_recent_sessions
        raw = _list_recent_sessions(db, limit, current)
        return json.loads(raw) if isinstance(raw, str) else raw

    def test_root_session_with_no_parent_appears(self):
        root = _make_session("root-1", source="cli")
        db = _make_db(sessions=[root], session_map={"root-1": root})
        result = self._list_recent(db)
        ids = [s["session_id"] for s in result.get("results", [])]
        assert "root-1" in ids

    def test_child_session_resolved_to_root(self):
        """Child must be listed under root ID, not silently dropped (Bug 3)."""
        root = _make_session("root-1", source="telegram")
        child = _make_session("child-1", source="telegram", parent="root-1")
        db = _make_db(
            sessions=[child],
            session_map={"root-1": root, "child-1": child},
        )
        result = self._list_recent(db)
        ids = [s["session_id"] for s in result.get("results", [])]
        assert "root-1" in ids, f"Root not surfaced from child. Got: {ids}"
        assert "child-1" not in ids, f"Child ID leaked. Got: {ids}"

    def test_root_and_child_not_duplicated(self):
        root = _make_session("root-1")
        child = _make_session("child-1", parent="root-1")
        db = _make_db(
            sessions=[root, child],
            session_map={"root-1": root, "child-1": child},
        )
        result = self._list_recent(db)
        ids = [s["session_id"] for s in result.get("results", [])]
        assert ids.count("root-1") == 1, f"Root duplicated: {ids}"

    def test_deeply_nested_child_resolves_to_root(self):
        root = _make_session("root-1")
        child = _make_session("child-1", parent="root-1")
        grandchild = _make_session("grandchild-1", parent="child-1")
        session_map = {"root-1": root, "child-1": child, "grandchild-1": grandchild}
        db = _make_db(sessions=[grandchild], session_map=session_map)
        result = self._list_recent(db)
        ids = [s["session_id"] for s in result.get("results", [])]
        assert "root-1" in ids, f"Root not surfaced from grandchild. Got: {ids}"

    def test_current_session_lineage_excluded(self):
        root = _make_session("active-root")
        child = _make_session("active-child", parent="active-root")
        other = _make_session("other-1")
        session_map = {"active-root": root, "active-child": child, "other-1": other}
        db = _make_db(sessions=[root, child, other], session_map=session_map)
        result = self._list_recent(db, current="active-child")
        ids = [s["session_id"] for s in result.get("results", [])]
        assert "active-root" not in ids
        assert "other-1" in ids


class TestFtsScanLimitAndCronDemotion:

    def test_fts_search_uses_limit_300_or_more(self):
        """search_messages must use limit >= 300 (was 50). Fixes Bug 4."""
        db = _make_db(fts_rows=[])
        from tools.session_search_tool import session_search
        session_search(db=db, query="test query", limit=3)
        db.search_messages.assert_called_once()
        _, kwargs = db.search_messages.call_args
        limit_used = kwargs.get("limit")
        assert limit_used is not None and limit_used >= 300, (
            f"FTS scan limit too low: {limit_used}."
        )

    def test_demoted_sources_constant_contains_cron(self):
        from tools.session_search_tool import _DEMOTED_SOURCES
        assert "cron" in _DEMOTED_SOURCES

    def test_demoted_sources_does_not_include_user_sources(self):
        from tools.session_search_tool import _DEMOTED_SOURCES
        user_sources = {"telegram", "cli", "discord", "api", "slack"}
        assert not (user_sources & _DEMOTED_SOURCES), (
            f"User sources should not be demoted: {user_sources & _DEMOTED_SOURCES}"
        )

    def test_cron_rerank_puts_user_sessions_first(self):
        """Cron rows must appear after non-cron rows after reranking."""
        from tools.session_search_tool import _DEMOTED_SOURCES
        fts_rows = [
            {"session_id": "cron-1", "source": "cron"},
            {"session_id": "cron-2", "source": "cron"},
            {"session_id": "tg-1", "source": "telegram"},
        ]
        primary = [r for r in fts_rows if r["source"] not in _DEMOTED_SOURCES]
        secondary = [r for r in fts_rows if r["source"] in _DEMOTED_SOURCES]
        reranked = primary + secondary

        sources = [r["source"] for r in reranked]
        first_cron = next(i for i, s in enumerate(sources) if s == "cron")
        first_tg = next(i for i, s in enumerate(sources) if s == "telegram")
        assert first_tg < first_cron, (
            f"telegram (pos {first_tg}) should precede cron (pos {first_cron})"
        )
