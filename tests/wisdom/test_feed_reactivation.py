import json
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from hermes_cli.config import save_config
from hermes_wisdom.account_session import sign_out
from hermes_wisdom.client import WisdomClient, WisdomConflict, WisdomError
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore


@pytest.fixture(params=["current", "signed-out-v11", "signed-out-v10"])
def runtime(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    save_config({"wisdom": {"notifications": {"new_skills": "immediate"}}})
    state = {
        "org": "one",
        "user": "first",
        "events": [],
        "calls": [],
        "interrupt": None,
        "fail": None,
    }
    from tests.wisdom.local_auth import authorize_local

    authorize_local(monkeypatch, lambda: state["org"])
    identities = {}
    monkeypatch.setattr(
        "hermes_wisdom.client.resolve_identity",
        lambda: {"api_key": "fixture", "claims": {"org_id": state["org"]}},
    )
    monkeypatch.setattr(
        "hermes_wisdom.client.resolve_sync_base_url", lambda: "https://gateway.example"
    )
    store = WisdomStore()

    class Transport(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            route = urlparse(request.url).path.rsplit("/", 1)[-1]
            query = parse_qs(urlparse(request.url).query)
            state["calls"].append((route, query))
            status = 200
            if route == "capabilities":
                body = {"features": ["wisdom"]}
            elif route == "installation-identities":
                identity = json.loads(request.body)["installation_id"]
                owner = (state["org"], state["user"])
                if state["interrupt"] == "identity_revoked":
                    state["interrupt"] = None
                    status, body = 409, {"error": "identity_revoked"}
                elif identity in identities and identities[identity] != owner:
                    status, body = 409, {"error": "identity_conflict"}
                else:
                    identities[identity] = owner
                    body = {"installation_id": identity, "state": "active"}
            elif route == "feed":
                cursor = query.get("cursor", [None])[0]
                # Opaque transport cursors, deliberately not timestamps.
                position = int(cursor.split(":")[-1]) if cursor else 0
                assert not cursor or cursor.startswith(state["org"] + ":")
                page = state["events"][position : position + 1]
                body = {
                    "events": page,
                    "has_more": position + 1 < len(state["events"]),
                    "next_cursor": f"{state['org']}:{position + len(page)}",
                }
                if state["fail"] in {"http", "stalled"} and position == 2:
                    if state["fail"] == "http":
                        status, body = 503, {"error": "fixture_unavailable"}
                    else:
                        body = {"events": [], "next_cursor": cursor, "has_more": True}
                    state["fail"] = None
            else:
                raise AssertionError(
                    f"Unexpected request: {request.method} {request.url}"
                )
            if state["interrupt"] == route:
                state["interrupt"] = None
                sign_out(WisdomStore(store.root))
            response = requests.Response()
            response.status_code = status
            response._content = json.dumps(body).encode()
            return response

        def close(self):
            pass

    clients = []

    def service():
        client = WisdomClient()
        client.session.mount("https://gateway.example/", Transport())
        clients.append(client)
        return WisdomService(store=WisdomStore(store.root), client=client)

    def publish(identity):
        state["events"].append({
            "event_id": identity,
            "kind": "new",
            "skill_id": "shared-skill",
            "version": 1,
            "installation_id": None,
            "update_mode": None,
            "occurred_at": "2026-01-01T00:00:00+00:00",
        })

    def logout():
        sign_out(store)
        if request.param != "current":
            with store.transaction() as db:
                for column in ("organization_id", "installation_id", "resume_required"):
                    db.execute(f"ALTER TABLE feed_state DROP COLUMN {column}")
                if request.param == "signed-out-v10":
                    db.execute("ALTER TABLE feed_state DROP COLUMN generation")
                version = request.param.rsplit("v", 1)[-1]
                db.execute(
                    "UPDATE schema_meta SET value=? WHERE key='schema_version'",
                    (version,),
                )
            state["legacy"] = True

    yield store, state, service, publish, logout
    for client in clients:
        client.session.close()


@pytest.mark.parametrize("scope", ["same", "team", "account"])
@pytest.mark.parametrize("interrupted", [None, "http", "stalled", "limit"])
def test_reactivation_discards_signed_out_backlog_then_delivers_new_arrivals(
    runtime, scope, interrupted
):
    store, state, service, publish, logout = runtime
    publish("before")
    first = service()
    initial = first.setup(disclosure_accepted=True)
    assert not any(route == "feed" for route, _ in state["calls"])
    first.consumption.poll_feed()
    assert [row["event_id"] for row in store.feed_events(unseen_only=True)] == [
        "before"
    ]
    logout()
    if scope == "team":
        state["org"] = "two"
        state["events"] = []
        publish("other-team-history")
    elif scope == "account":
        state["user"] = "second"
    publish("during-one")
    publish("during-two")
    if interrupted == "limit":
        for index in range(103):
            publish(f"backlog-{index}")
    state["calls"].clear()
    state["fail"] = interrupted
    if interrupted:
        message = {"http": "503", "stalled": "no progress", "limit": "page limit"}[
            interrupted
        ]
        with pytest.raises(WisdomError, match=message):
            service().setup(disclosure_accepted=True)
        assert store.active_org_id() is None
        resume_cursor = store.feed_cursor()
        assert resume_cursor.startswith(state["org"] + ":")
        assert 0 < int(resume_cursor.split(":")[-1]) < len(state["events"])
        state["calls"].clear()
    resumed = service()
    result = resumed.setup(disclosure_accepted=True)
    feed_calls = [query for route, query in state["calls"] if route == "feed"]
    assert feed_calls
    expected_start = (
        resume_cursor
        if interrupted
        else ("one:1" if scope == "same" and not state.get("legacy") else None)
    )
    assert feed_calls[0].get("cursor", [None])[0] == expected_start
    assert store.feed_cursor() == f"{state['org']}:{len(state['events'])}"
    assert store.feed_events(unseen_only=True) == []
    assert (result["installation_id"] != initial["installation_id"]) == (
        scope != "same"
    )
    publish("after")
    resumed.consumption.poll_feed()
    assert [row["event_id"] for row in store.feed_events(unseen_only=True)] == ["after"]


@pytest.mark.parametrize(
    "phase", ["capabilities", "installation-identities", "feed", "identity_revoked"]
)
def test_logout_during_reactivation_cannot_restore_authority(runtime, phase):
    store, state, service, publish, logout = runtime
    service().setup(disclosure_accepted=True)
    logout()
    publish("during")
    state["interrupt"] = phase
    state["calls"].clear()
    with pytest.raises(
        WisdomConflict,
        match="identity_revoked" if phase == "identity_revoked" else "account changed",
    ):
        service().setup(disclosure_accepted=True)
    if phase == "identity_revoked":
        assert [route for route, _ in state["calls"]].count(
            "installation-identities"
        ) == 1
    assert store.active_org_id() is None
    assert store.feed_events(unseen_only=True) == []
    service().setup(disclosure_accepted=True)
    assert store.active_org_id() == "one"
    assert store.feed_events(unseen_only=True) == []
