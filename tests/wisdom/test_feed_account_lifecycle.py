import json
import sqlite3
from contextlib import closing

import pytest
import requests

from hermes_wisdom.account_session import sign_out
from hermes_wisdom.client import WisdomAuthError, WisdomClient, WisdomConflict
from hermes_wisdom.consumption import WisdomConsumption
from hermes_wisdom.store import WisdomStore


@pytest.fixture(params=["fresh", "legacy"])
def feed(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    monkeypatch.setattr(
        "hermes_wisdom.client.resolve_identity",
        lambda: {"api_key": "fixture", "claims": {"org_id": "org"}},
    )
    monkeypatch.setattr(
        "hermes_wisdom.client.resolve_sync_base_url", lambda: "https://gateway.example"
    )
    if request.param == "legacy":
        root = tmp_path / "profile" / "wisdom"
        root.mkdir(parents=True)
        with closing(sqlite3.connect(root / "wisdom.db")) as db:
            db.execute(
                "CREATE TABLE feed_state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                "cursor TEXT,updated_at TEXT NOT NULL)"
            )
            db.execute("INSERT INTO feed_state VALUES(1,'legacy-cursor','2026-01-01')")
            db.commit()
    store = WisdomStore()
    if request.param == "legacy":
        assert store.feed_cursor() == "legacy-cursor"
    store.activate_installation_identity("installation", "org")
    client = WisdomClient()
    manager = WisdomConsumption(
        store=store, client=client, scan=lambda _: {}, config={}
    )
    yield store, client, manager
    client.session.close()


def event(identity):
    return {
        "event_id": identity,
        "kind": "new",
        "skill_id": "shared-skill",
        "version": 1,
        "installation_id": None,
        "update_mode": None,
        "occurred_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.mark.parametrize("surface", ["local", "telegram", "slack"])
def test_logout_retires_cached_unassessed_notices_without_erasing_history(
    feed, surface
):
    store, _, _ = feed
    store.persist_feed_page(
        [event("cached")],
        next_cursor="before",
        cadences={},
        now="2026-01-01T00:00:00+00:00",
    )
    store.mark_feed_surface_delivered(["cached"], surface="desktop")
    sign_out(store)
    store.verify_installation_identity("org")
    restarted = WisdomStore(store.root)

    due = (
        restarted.feed_events(unseen_only=True)
        if surface == "local"
        else restarted.feed_events(
            surface=surface, surface_due_at="2999-01-01T00:00:00+00:00"
        )
    )
    assert due == []
    assert [row["event_id"] for row in restarted.feed_events()] == ["cached"]
    assert restarted.feed_cursor() == "before"
    with restarted.transaction() as db:
        assert (
            db.execute(
                "SELECT surface FROM feed_event_delivery WHERE event_id='cached'"
            ).fetchone()[0]
            == "desktop"
        )


@pytest.mark.parametrize("reverify", [False, True])
def test_feed_response_cannot_cross_logout_even_after_same_team_reverification(
    feed, reverify
):
    store, client, manager = feed
    store.persist_feed_page(
        [], next_cursor="before", cadences={}, now="2026-01-01T00:00:00+00:00"
    )

    class ReturningAfterLogout(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            assert "cursor=before" in request.url
            sign_out(WisdomStore(store.root))
            if reverify:
                store.verify_installation_identity("org")
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({
                "events": [event("late")],
                "next_cursor": "after",
                "has_more": False,
            }).encode()
            return response

        def close(self):
            pass

    client.session.mount("https://gateway.example/", ReturningAfterLogout())
    with pytest.raises(WisdomConflict, match="account changed"):
        manager.poll_feed()
    assert store.feed_events() == []
    assert store.feed_cursor() == "before"
    if not reverify:
        with pytest.raises(WisdomAuthError, match="sign in"):
            manager.poll_feed()
