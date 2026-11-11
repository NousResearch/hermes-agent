"""The model picker of a served profile fetches provider pricing with that profile's own key and endpoint.

One ``hermes serve`` / Desktop backend hosts several profiles. ``os.environ`` holds the launch profile's
``.env``; a picker refresh for profile B must never send the launch profile's provider key, or ask the
launch profile's endpoint, on B's behalf.
"""

from __future__ import annotations

import pytest

from tests.fakes.providers.catalog_fake import CatalogFake
from tui_gateway import launch_profile_policy as lpp
from tui_gateway import server


@pytest.fixture
def two_homes_and_endpoints(tmp_path, monkeypatch):
    with CatalogFake(routes={"/launch/v1/models": "listing", "/second/v1/models": "listing"}) as fake:
        root = tmp_path / "hermes_home"
        second = root / "profiles" / "second"
        second.mkdir(parents=True)
        launch_env = {"NOVITA_API_KEY": "sk-launch-novita", "NOVITA_BASE_URL": f"{fake.origin}/launch/v1"}
        (root / ".env").write_text("".join(f"{k}={v}\n" for k, v in launch_env.items()), encoding="utf-8")
        (second / ".env").write_text(
            f"NOVITA_API_KEY=sk-second-novita\nNOVITA_BASE_URL={fake.origin}/second/v1\n", encoding="utf-8")
        for home in (root, second):
            (home / "config.yaml").write_text("model:\n  provider: novita\n  default: m\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(root))
        for name, value in launch_env.items():
            monkeypatch.setenv(name, value)  # the launch process loaded its own .env
        monkeypatch.setattr(server, "_hermes_home", root)
        monkeypatch.setattr(server, "_served_profile_homes", set())
        monkeypatch.setattr(lpp, "_snapshot", None)
        from agent import secret_scope
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        yield fake


def test_picker_refresh_for_a_served_profile_never_uses_the_launch_profile_key(two_homes_and_endpoints):
    fake = two_homes_and_endpoints

    response = server._methods["model.options"]("rid", {"profile": "second", "refresh": True})

    assert "error" not in response, response
    sent = [(r.path, r.headers.get("Authorization") or r.headers.get("authorization")) for r in fake.requests]
    assert ("/second/v1/models", "Bearer sk-second-novita") in sent
    assert not [s for s in sent if s[0].startswith("/launch/") or s[1] == "Bearer sk-launch-novita"], sent
