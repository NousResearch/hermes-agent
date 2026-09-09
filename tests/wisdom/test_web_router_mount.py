"""Exercise the mounted API without starting any demo or background services."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server, web_server_profiles
from hermes_cli.web_routers import wisdom as wisdom_routes
from hermes_constants import get_hermes_home


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    # No lifespan context: startup would launch unrelated gateway services.
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    client.close()


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/wisdom/status"),
    ("GET", "/api/wisdom/sync"),
    ("POST", "/api/wisdom/sync/retry"),
    ("GET", "/api/wisdom/candidates"),
    ("POST", "/api/wisdom/setup"),
    ("POST", "/api/wisdom/publication/review"),
    ("POST", "/api/wisdom/publication/submit"),
    ("POST", "/api/wisdom/consent"),
    ("POST", "/api/wisdom/mute/choose"),
    ("POST", "/api/wisdom/install/apply"),
    ("POST", "/api/wisdom/update/apply"),
])
def test_wisdom_routes_require_dashboard_auth(client, monkeypatch, method, path):
    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthenticated request reached Wisdom")

    monkeypatch.setattr(wisdom_routes, "_run_wisdom", forbidden)
    assert client.request(method, path).status_code == 401


def test_mounted_review_and_final_confirmation_preserve_package_and_policy(client, monkeypatch):
    calls = []

    class Service:
        def publication_review(self, draft_id):
            calls.append(("review", draft_id))
            return {"publication_mode": "moderated"}

        def submit_reviewed_package(self, draft_id, **kwargs):
            calls.append(("submit", draft_id, kwargs))
            return {"publication_state": "pending_moderation"}

    async def run(profile, fn):
        assert profile == "research"
        return fn(Service())

    monkeypatch.setattr(wisdom_routes, "_run_wisdom", run)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    review = {"draft_id": "local:review-1", "profile": "research"}
    response = client.post("/api/wisdom/publication/review", json=review)
    assert response.status_code == 200
    assert response.json() == {"publication_mode": "moderated"}
    assert calls == [("review", "local:review-1")]

    hashes = {"content": "content", "author_description": "copy", "package_manifest": "manifest"}
    body = {**review, "expected_hashes": hashes, "publication_mode": "moderated"}
    for invalid in (
        {**body, "expected_hashes": {"content": "content"}},
        {**body, "session_id": "session-1"},
        {**body, "interaction_id": "interaction-1"},
        {**body, "publication_mode": "skip_checks"},
    ):
        assert client.post("/api/wisdom/publication/submit", json=invalid).status_code == 422
    assert calls == [("review", "local:review-1")]

    response = client.post("/api/wisdom/publication/submit", json=body)
    assert response.status_code == 200
    assert response.json() == {"publication_state": "pending_moderation"}
    assert calls[-1] == ("submit", "local:review-1", {
        "expected_hashes": hashes, "publication_mode": "moderated",
    })


def test_profile_scope_is_preserved_through_real_worker_dispatch(client, monkeypatch, tmp_path):
    homes = {name: tmp_path / name for name in ("research", "personal")}
    for home in homes.values():
        home.mkdir()
    baseline = get_hermes_home()

    class Service:
        def status(self):
            return {"home": str(get_hermes_home())}

    monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", homes.__getitem__)
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Service)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    for name, home in homes.items():
        response = client.get("/api/wisdom/status", params={"profile": name})
        assert response.status_code == 200
        assert response.json() == {"home": str(home)}
    assert get_hermes_home() == baseline

    async def concurrent_reads():
        return await asyncio.gather(*(wisdom_routes.get_wisdom_status(name) for name in homes))

    assert asyncio.run(concurrent_reads()) == [{"home": str(home)} for home in homes.values()]
    assert get_hermes_home() == baseline
