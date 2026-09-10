"""The conversational session-search tool must opt into NL expansion."""

from tools import session_search_tool


class _DB:
    def __init__(self):
        self.kwargs = None

    def search_messages(self, **kwargs):
        self.kwargs = kwargs
        return []

    def list_sessions(self, **kwargs):
        return []

    def fts_rebuild_status(self):
        return None


def test_discover_enables_natural_language_search():
    db = _DB()
    result = session_search_tool._discover(
        db=db,
        query="what did we decide about the backups?",
        role_filter=None,
        limit=3,
        sort=None,
        detail="adaptive",
    )

    assert db.kwargs is not None
    assert db.kwargs["natural_language"] is True
    assert '"count": 0' in result
