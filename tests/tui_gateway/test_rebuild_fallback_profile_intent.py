"""Rebuild must reconcile profile intent before publishing config_model_seen.

Real construction/admission and SQLite, inert clients, isolated owning profiles.
"""
from pathlib import Path

import pytest

from tests.tui_gateway.test_fallback_recovery_boundaries import (
    FALLBACK, PRIMARY, _build, _admit, _request_fallback, runtime_env,
)
from tui_gateway import server


@pytest.mark.parametrize("origin", ["construction", "request"])
def test_profile_adoption_survives_rebuild_before_next_turn(monkeypatch, runtime_env, origin):
    clock, db, resolved, home = runtime_env
    clock["auth_failed"] = origin == "construction"
    original = _build(home, db, "sid", "key")
    if origin == "request":
        _request_fallback(monkeypatch, original, home)
    db.create_session("key", source="desktop", model=PRIMARY,
                      model_config={"follow_profile_config": True})
    session = {"agent": original, "session_key": "key", "profile_home": home,
               "follow_profile_config": True, "config_model_seen": (PRIMARY, "anthropic")}
    owner = Path(home or server._hermes_home)
    (owner / "config.yaml").write_text(
        f"model:\n  default: {FALLBACK}\n  provider: openai-codex\n")
    clock.update(wall=1010.0, mono=110.0, auth_failed=False)
    rebuilt = server._rebuild_session_agent("sid", session)
    try:
        assert rebuilt.model == FALLBACK
        clock.update(wall=1200.0, mono=300.0)
        _admit(monkeypatch, session, real_config_sync=True)
        assert rebuilt.model == FALLBACK, (
            "Rebuild marked the edited profile seen while retaining obsolete primary recovery; "
            f"expected {FALLBACK}, got {rebuilt.model}; seen={session['config_model_seen']}"
        )
        assert getattr(rebuilt, "_tui_fallback_recovery", None) is None
    finally:
        original.close()
        rebuilt.close()


@pytest.mark.parametrize("origin", ["construction", "request"])
@pytest.mark.parametrize("change", [
    "unchanged", "pin", "model", "provider", "base_url", "api_mode",
    "remove_endpoint", "remove_wire",
])
def test_rebuild_intent_and_cooldown_controls(monkeypatch, runtime_env, origin, change):
    import json
    import hermes_yaml as yaml
    from tui_gateway.fallback_recovery import recovery_state

    clock, db, resolved, home = runtime_env
    from hermes_cli import runtime_provider
    inert_resolve = runtime_provider.resolve_runtime_provider

    def resolve_with_profile_default(**kwargs):
        # The shared inert fixture defaults to Anthropic; production resolution
        # reads the owning profile when startup passes no explicit provider.
        if not kwargs.get("requested"):
            kwargs["requested"] = server._load_cfg()["model"]["provider"]
        return inert_resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve_with_profile_default)
    owner = Path(home or server._hermes_home)
    config_path = owner / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    if change == "remove_endpoint":
        config["model"]["base_url"] = "https://old.example.invalid/v1"
    if change == "remove_wire":
        config["model"]["api_mode"] = "anthropic_messages"
    config_path.write_text(yaml.safe_dump(config))
    pin = {"model": PRIMARY, "provider": "anthropic"}
    clock["auth_failed"] = origin == "construction"
    original = _build(home, db, "sid", "key", **(
        {"model_override": pin} if change == "pin" else {}))
    if origin == "request":
        _request_fallback(monkeypatch, original, home)
    prior = recovery_state(original)
    assert prior is not None
    db.create_session("key", source="desktop", model=PRIMARY,
                      model_config={"follow_profile_config": True})
    session = {"agent": original, "session_key": "key", "profile_home": home,
               "follow_profile_config": True, "config_model_seen": (PRIMARY, "anthropic")}
    if change == "pin":
        session["model_override"] = pin
        config["model"] = {"default": FALLBACK, "provider": "openai-codex"}
    elif change == "model":
        config["model"]["default"] = "claude-sonnet-5-5"
    elif change == "provider":
        config["model"]["provider"] = "openrouter"
    elif change == "base_url":
        config["model"]["base_url"] = "https://new.example.invalid/v1"
    elif change == "api_mode":
        config["model"]["api_mode"] = "anthropic_messages"
    elif change == "remove_endpoint":
        del config["model"]["base_url"]
    elif change == "remove_wire":
        del config["model"]["api_mode"]
    config_path.write_text(yaml.safe_dump(config))
    clock.update(wall=1010.0, mono=110.0, auth_failed=False)
    rebuilt = server._rebuild_session_agent("sid", session)
    try:
        held = change in ("unchanged", "pin")
        if held:
            assert rebuilt.model == FALLBACK
            assert recovery_state(rebuilt) == prior
            calls = len(resolved)
            _admit(monkeypatch, session, real_config_sync=True)
            assert len(resolved) == calls
            assert recovery_state(rebuilt) == prior
        else:
            assert recovery_state(rebuilt) is None
            assert rebuilt.model == config["model"]["default"]
            assert rebuilt.provider == config["model"]["provider"]
            assert rebuilt.base_url == config["model"].get("base_url", "https://example.invalid")
            assert rebuilt.api_mode == config["model"].get("api_mode", "chat_completions")
        clock.update(wall=1200.0, mono=300.0)
        _admit(monkeypatch, session, real_config_sync=True)
        assert recovery_state(rebuilt) is None
        assert rebuilt.model == (PRIMARY if held else config["model"]["default"])
        assert rebuilt.provider == ("anthropic" if held else config["model"]["provider"])
        with server._profile_build_scope(home):
            server._persist_live_session_runtime(session)
        row = db.get_session("key")
        assert row["model"] == rebuilt.model
        assert "fallback_recovery" not in json.loads(row["model_config"])
        if change == "pin":
            assert session["model_override"] == pin
        if home:
            assert {scope for scope, _ in resolved} == {str(home)}
    finally:
        original.close()
        rebuilt.close()
