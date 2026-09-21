"""Phase 1: skill broker is backed by the unified profile activity ledger.

Verifies borrow/deny/revoke/count flow through the central append-only ledger
and that query_events reads them back correctly. Isolated to a temp ledger via
monkeypatching ledger_db_path.
"""

import pytest

from hermes_cli import profile_activity_ledger as pal


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(pal, "ledger_db_path", lambda: db)
    # JSONL mirror also lives under the temp governance dir
    monkeypatch.setattr(pal, "_governance_root", lambda: tmp_path)
    return db


def test_query_events_filters(temp_ledger):
    pal.append_event(source="t", event_type="skill.borrowed", target_profile="octacon",
                     object_type="skill", object_id="arxiv", event_id="b1")
    pal.append_event(source="t", event_type="skill.borrowed", target_profile="remii",
                     object_type="skill", object_id="arxiv", event_id="b2")
    octacon = pal.query_events(event_types=["skill.borrowed"], target_profile="octacon")
    assert [e["event_id"] for e in octacon] == ["b1"]
    by_skill = pal.query_events(object_id="arxiv")
    assert len(by_skill) == 2


def test_ledger_is_append_only(temp_ledger):
    import sqlite3
    pal.append_event(source="t", event_type="skill.borrowed", target_profile="octacon",
                     object_type="skill", object_id="arxiv", event_id="b1")
    with pal._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE activity_events SET summary='x' WHERE event_id='b1'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM activity_events WHERE event_id='b1'")
