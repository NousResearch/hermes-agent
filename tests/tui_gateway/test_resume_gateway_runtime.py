"""Resume a gateway-written route without borrowing the first billing provider."""
from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin
from hermes_cli.cli_model_switch_mixin import stored_session_route
from hermes_state import SessionDB
from tui_gateway.server import _stored_session_runtime_overrides


def test_resume_gateway_route_matches_cli_after_first_billing_call(tmp_path):
    from hermes_cli.providers import resolve_provider_full
    from hermes_cli.config import load_config
    config = load_config()
    assert resolve_provider_full("anthropic", config.get("providers")) is not None
    assert resolve_provider_full("openai-codex", config.get("providers")) is not None
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("route", source="telegram", model="old-model")
        db.update_token_counts("route", input_tokens=1, output_tokens=1,
                               model="old-model", billing_provider="anthropic", api_call_count=1)
        gateway = SimpleNamespace(_session_db=SimpleNamespace(_db=db))
        agent = SimpleNamespace(model="current-model", provider="openai-codex",
                                base_url=None, api_mode="codex_responses", _fallback_activated=False)
        GatewayTurnMixin._sync_session_model_from_agent(gateway, "route", agent)
        row = db.get_session("route")
        assert row["billing_provider"] == "anthropic"
        cli = stored_session_route(row, current_model="old-model", current_provider="anthropic")
        assert cli[:4] == (agent.model, agent.provider, None, agent.api_mode)
        restored = _stored_session_runtime_overrides(row)["model_override"]
        assert (restored["model"], restored["provider"], restored["base_url"], restored["api_mode"]) == cli[:4]
    finally:
        db.close()


@pytest.mark.parametrize("runtime", [None, {}, [], {"api_mode": "codex_responses"},
                                     {"provider": "openai-codex", "fallback_active": True}])
def test_incomplete_or_fallback_gateway_route_preserves_billing_behavior(runtime):
    row = {"model": "old-model", "billing_provider": "anthropic",
           "model_config": {"gateway_runtime": runtime}}
    assert _stored_session_runtime_overrides(row)["model_override"]["provider"] == "anthropic"


def test_explicit_desktop_pick_and_settings_survive_old_gateway_snapshot():
    row = {"model": "picked-model", "billing_provider": "openai-codex", "model_config": {
        "provider": "anthropic", "reasoning_config": {"enabled": False}, "service_tier": "normal",
        "gateway_runtime": {"provider": "openai-codex", "api_mode": "codex_responses"}}}
    restored = _stored_session_runtime_overrides(row)
    assert restored["model_override"]["provider"] == "anthropic"
    assert restored["reasoning_config_override"] == {"enabled": False}
    assert restored["service_tier_override"] == ""


@pytest.mark.parametrize("row", [None, {}, {"model": "m", "model_config": "malformed"}])
def test_missing_or_invalid_metadata_does_not_restore_provider(row):
    restored = _stored_session_runtime_overrides(row)
    assert restored.get("model_override", {}).get("provider") is None


@pytest.mark.parametrize("provider", ["some-dead-provider", "auto", "custom", "  "])
def test_unusable_gateway_route_preserves_session_endpoint(provider, tmp_path):
    import json

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("stale-route", source="telegram", model="m1")
        config = {
            "base_url": "https://myproxy.example/v1", "api_mode": "chat_completions",
            "gateway_runtime": {"provider": provider, "api_mode": "codex_responses"},
        }
        db.update_session_meta("stale-route", json.dumps(config), "m1")
        from hermes_constants import get_hermes_home
        from tui_gateway.server import _resolve_agent_model_runtime
        import yaml

        # A real configured default differs from the row's endpoint. No provider
        # request is sent: only the resume/build resolution chain runs.
        (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({
            "model": {"default": "m1", "provider": "custom:fixture"},
            "custom_providers": [{"name": "fixture", "base_url": "https://default.invalid/v1",
                                  "api_key": "fixture-not-a-real-key", "api_mode": "chat_completions"}],
        }), encoding="utf-8")
        restored = _stored_session_runtime_overrides(db.get_session("stale-route"))
        assert restored["model_override"] == {
            "model": "m1", "provider": None,
            "base_url": config["base_url"], "api_mode": config["api_mode"],
        }
        model, runtime = _resolve_agent_model_runtime(
            restored["model_override"], restored.get("provider_override"))
        assert model == "m1"
        assert runtime["base_url"] == config["base_url"]
        assert runtime["api_mode"] == config["api_mode"]
    finally:
        db.close()


@pytest.mark.parametrize("provider", ["some-dead-provider", "auto", "custom", "  "])
def test_unusable_gateway_route_preserves_billing_fallback(provider):
    row = {"model": "m1", "billing_provider": "anthropic", "model_config": {
        "gateway_runtime": {"provider": provider, "api_mode": "codex_responses"}}}
    restored = _stored_session_runtime_overrides(row)["model_override"]
    assert (restored["provider"], restored["base_url"], restored["api_mode"]) == (
        "anthropic", None, None)


def test_gateway_endpoint_is_not_combined_with_older_top_level_endpoint():
    row = {"model": "current-model", "billing_provider": "anthropic", "model_config": {
        "base_url": "https://old.invalid/v1", "api_mode": "chat_completions",
        "gateway_runtime": {"provider": "openai-codex", "base_url": None, "api_mode": "codex_responses"}}}
    restored = _stored_session_runtime_overrides(row)["model_override"]
    assert (restored["provider"], restored["base_url"], restored["api_mode"]) == (
        "openai-codex", None, "codex_responses")


@pytest.mark.parametrize("config", [{"follow_profile_config": True}, {"room_plumbing": True}])
def test_profile_following_sessions_keep_their_exemption(config):
    config["gateway_runtime"] = {"provider": "openai-codex"}
    assert _stored_session_runtime_overrides({"model": "current-model", "model_config": config}) == {}
