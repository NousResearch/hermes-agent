"""Tests for the low-token session_search response shape."""

import json

from tools.session_search_tool import session_search


class FakeSessionDB:
    def __init__(self):
        self.anchored_calls = []
        self.scroll_calls = []

    def search_messages(self, **kwargs):
        return [{
            "id": 42,
            "session_id": "s1",
            "role": "assistant",
            "snippet": "matched <b>operation</b> summary",
            "source": "feishu",
            "model": "gemini-3-flash",
            "session_started": 1000,
        }]

    def get_session(self, session_id):
        return {
            "id": session_id,
            "title": "Morning briefing fix",
            "source": "feishu",
            "model": "gemini-3-flash",
            "started_at": 1000,
        }

    def get_anchored_view(self, session_id, message_id, window, bookend):
        self.anchored_calls.append((session_id, message_id, window, bookend))
        return {
            "bookend_start": [{"id": 1, "role": "user", "content": "start"}],
            "window": [{"id": message_id, "role": "assistant", "content": "match"}],
            "bookend_end": [{"id": 99, "role": "assistant", "content": "end"}],
            "messages_before": 10,
            "messages_after": 3,
        }

    def get_messages_around(self, session_id, around_message_id, window):
        self.scroll_calls.append((session_id, around_message_id, window))
        return {
            "window": [{"id": around_message_id, "role": "assistant", "content": "match"}],
            "messages_before": 1,
            "messages_after": 1,
        }


def test_discovery_defaults_to_summary_without_messages_or_bookends():
    db = FakeSessionDB()

    result = json.loads(session_search(query="operation summary", db=db))

    assert result["success"] is True
    assert result["mode"] == "discover"
    assert result["detail"] == "summary"
    entry = result["results"][0]
    assert entry["match_message_id"] == 42
    assert entry["snippet"] == "matched <b>operation</b> summary"
    assert "scroll_hint" in entry
    assert "messages" not in entry
    assert "bookend_start" not in entry
    assert "bookend_end" not in entry
    assert db.anchored_calls == []


def test_discovery_full_detail_keeps_complete_shape_with_small_window():
    db = FakeSessionDB()

    result = json.loads(session_search(query="operation summary", detail="full", db=db))

    assert result["detail"] == "full"
    entry = result["results"][0]
    assert entry["messages"][0]["anchor"] is True
    assert entry["bookend_start"][0]["content"] == "start"
    assert entry["bookend_end"][0]["content"] == "end"
    assert db.anchored_calls == [("s1", 42, 1, 1)]


def test_scroll_default_window_is_two_and_explicit_window_still_works():
    db = FakeSessionDB()

    result = json.loads(session_search(session_id="s1", around_message_id=42, db=db))
    assert result["window"] == 2
    assert db.scroll_calls[-1] == ("s1", 42, 2)

    result = json.loads(
        session_search(session_id="s1", around_message_id=42, window=7, db=db)
    )
    assert result["window"] == 7
    assert db.scroll_calls[-1] == ("s1", 42, 7)
