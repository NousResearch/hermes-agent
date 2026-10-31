"""``_session_by_title`` exact-first title lookup with a case/whitespace-insensitive fallback.

SQLite's ``WHERE s.title = ?`` is case-sensitive, so ``session.list {title}`` and
``/resume <title>`` used to 4007 against a stored title that differed only in case or
spacing (``Navigate To Semester 5`` vs stored ``Navigate to Semester 5``) even though
``idx_sessions_title_unique`` guarantees an unambiguous normalized match.

Pinned here:

* exact match wins before the fallback ever runs (title stays identity);
* a normalized case/whitespace variant resolves to the FULL row re-fetched by id;
* no match, or an empty/whitespace title, returns ``None``;
* a listing-shape drift (``list_sessions_rich`` raising) degrades to the exact-only
  behaviour instead of propagating — never a 5000.
"""

from __future__ import annotations

from tui_gateway.methods_session import _session_by_title


class _DB:
    """Minimal ``SessionDB`` stand-in driven by an explicit row table."""

    def __init__(self, rows, exact=None, raise_on_list=False):
        self._rows = rows
        self._exact = exact
        self._raise = raise_on_list
        self.list_calls = 0

    def get_session_by_title(self, title):
        if self._exact is not None:
            return self._exact
        for row in self._rows:
            if row["title"] == title:
                return row
        return None

    def list_sessions_rich(self, **kwargs):
        self.list_calls += 1
        if self._raise:
            raise RuntimeError("listing shape drifted after `hermes update`")
        return list(self._rows)

    def get_session(self, session_id):
        for row in self._rows:
            if row["id"] == session_id:
                return row
        return None


def _row(sid, title):
    return {"id": sid, "title": title, "archived": False}


def test_exact_match_wins_without_fallback():
    db = _DB([_row("s1", "Navigate to Semester 5")])
    row = _session_by_title(db, "Navigate to Semester 5")
    assert row == _row("s1", "Navigate to Semester 5")
    assert db.list_calls == 0, "exact hit must not run the fallback listing"


def test_case_variant_resolves_to_full_row():
    db = _DB([_row("s1", "Navigate to Semester 5")])
    row = _session_by_title(db, "Navigate To Semester 5")
    assert row == _row("s1", "Navigate to Semester 5")


def test_whitespace_variant_resolves():
    db = _DB([_row("s1", "Navigate to  Semester 5")])
    row = _session_by_title(db, "Navigate  to Semester 5")
    assert row is not None
    assert row["id"] == "s1"


def test_missing_title_returns_none():
    db = _DB([_row("s1", "some other session")])
    assert _session_by_title(db, "Does Not Exist") is None


def test_empty_title_returns_none_without_fallback():
    db = _DB([_row("s1", "Navigate to Semester 5")])
    assert _session_by_title(db, "   ") is None
    assert db.list_calls == 0, "empty normalized title must skip the fallback"


def test_listing_drift_degrades_to_exact_only():
    db = _DB([_row("s1", "Navigate to Semester 5")], raise_on_list=True)
    # exact miss + broken listing -> graceful None (degrades to old behaviour)
    assert _session_by_title(db, "Navigate To Semester 5") is None
    # exact hit still succeeds even when the listing is broken
    db_exact = _DB([], exact=_row("s2", "Bot Chat"), raise_on_list=True)
    assert _session_by_title(db_exact, "Bot Chat") == _row("s2", "Bot Chat")
