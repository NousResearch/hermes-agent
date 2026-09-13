"""A local selection must survive session.info, a new chat and stored-session resume."""
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.local_runtime import endpoint
from tui_gateway import server


@pytest.fixture
def local_route(tmp_path, monkeypatch):
    cfg = {"model": {"provider": "anthropic", "default": "claude-test"},
           "local_runtime": {"enabled": True}}
    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    monkeypatch.setattr(rp, "_get_model_config", lambda: cfg["model"])
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    route = {"base_url": "http://127.0.0.1:18434/v1", "api_key": "local-test-key"}
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: route)
    monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **kw: route)
    monkeypatch.setattr(server, "_probe_credentials", lambda agent: None)
    monkeypatch.setattr("hermes_cli.banner.get_update_result", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.banner.get_available_skills", lambda: {})
    return route, {"cwd": str(tmp_path), "session_key": "local-identity"}


def test_live_local_identity_survives_new_chat_and_resume(local_route):
    route, session = local_route
    model = "Local.Model-Q4_K_M"
    runtime = rp.resolve_runtime_provider(requested="llamacpp", target_model=model)
    agent = SimpleNamespace(model=model, provider=runtime["provider"], base_url=runtime["base_url"],
                            api_mode=runtime["api_mode"], reasoning_config=None, service_tier=None,
                            session_id=session["session_key"])
    # The renderer carries these two fields into the next session.create.
    info = server._session_info(agent, session)
    assert info["provider"] == "llamacpp"
    assert info["model"] == model
    next_model, next_runtime = server._resolve_agent_model_runtime(
        {"model": info["model"], "provider": info["provider"]}, None)
    assert next_model == model and next_runtime["base_url"] == route["base_url"]
    assert next_runtime["api_key"] == route["api_key"]
    persisted = server._runtime_model_config(agent)
    assert persisted["provider"] == "llamacpp"
    assert "api_key" not in persisted
    # Legacy rows kept the local endpoint but lost the provider slug.
    for provider in ("custom", "llamacpp"):
        row = {"model": model, "model_config": {**persisted, "provider": provider}}
        overrides = server._stored_session_runtime_overrides(row)
        restored_model, restored = server._resolve_agent_model_runtime(
            overrides["model_override"], overrides.get("provider_override"))
        assert restored_model == model
        assert restored["base_url"] == route["base_url"]
        assert restored["api_key"] == route["api_key"]
    # Pending picks and compute-host mirrors still own the reported identity.
    session["pending_model_switch"] = {"display_model": "claude-test", "display_provider": "anthropic"}
    assert server._session_info(agent, session)["provider"] == "anthropic"


def test_session_info_recovers_identity_from_the_owning_profile(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from hermes_constants import get_hermes_home

    launch = tmp_path / "launch"
    secondary = launch / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    url = "https://session-endpoint.invalid/v1"
    for home, name in ((launch, "launch-route"), (secondary, "secondary-route")):
        config = {"model": {"provider": "anthropic", "default": "claude-test"},
                  "providers": {name: {"api": url, "models": ["same-model"]}}}
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_probe_credentials", lambda agent: None)
    monkeypatch.setattr("hermes_cli.banner.get_update_result", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.banner.get_available_skills", lambda: {})
    agent = SimpleNamespace(model="same-model", provider="custom", base_url=url,
                            reasoning_config=None, service_tier=None, session_id="profile-identity")
    session = {"cwd": str(tmp_path), "session_key": "profile-identity", "profile_home": str(secondary)}
    # Broadcast/resume can publish metadata outside the session's profile scope.
    assert server._session_info(agent, session)["provider"] == "custom:secondary-route"
    assert get_hermes_home() == launch
    # A launch-profile session must also ignore an ambient secondary-profile scope.
    with server._profile_build_scope(secondary):
        assert server._session_info(agent, {**session, "profile_home": None})["provider"] == "custom:launch-route"
        assert get_hermes_home() == secondary
    assert get_hermes_home() == launch
    # Remote compute metadata remains authoritative; never reinterpret it using local profiles.
    session["_metadata_mirror"] = {"model": "remote-model", "provider": "custom:remote-route"}
    assert server._session_info(agent, session)["provider"] == "custom:remote-route"


def test_local_identity_never_claims_an_unrelated_endpoint(local_route):
    route, _ = local_route
    assert rp.canonical_custom_identity(base_url=route["base_url"]) == "llamacpp"
    assert rp.canonical_custom_identity(base_url="http://127.0.0.1:18435/v1") is None
    assert rp.canonical_custom_identity(base_url="https://api.anthropic.com") is None
    assert rp.canonical_custom_identity(model="Local.Model-Q4_K_M") is None
