"""Untrusted persisted recovery metadata is optional, bounded and secret-free."""
import json

import pytest

from tui_gateway import server
from tui_gateway.fallback_recovery import valid_recovery


def _state():
    return {"primary": {"model": "primary", "provider": "anthropic"},
            "fallback": {"model": "fallback", "provider": "openai-codex"},
            "retry_at": 1234.5}


@pytest.mark.parametrize("deadline", [None, True, "1234", [], {}, float("nan"), float("inf"), 10 ** 400])
def test_invalid_deadline_does_not_break_session_resume(deadline):
    state = _state()
    state["retry_at"] = deadline
    row = {"model": "manual", "model_config": json.dumps({"fallback_recovery": state})}
    assert valid_recovery(state) is None
    assert server._stored_session_runtime_overrides(row)["model_override"]["model"] == "manual"


def test_recovery_only_retains_route_fields_and_finite_deadline():
    state = _state()
    for key in ("primary", "fallback"):
        state[key].update(api_key="secret", access_token="secret", credential_pool={"secret": True})
    cleaned = valid_recovery(state)
    assert cleaned == _state()
    assert "secret" not in json.dumps(cleaned)
    for bad in (None, [], {"primary": "bad"}, {**state, "fallback": state["primary"]},
                {**state, "primary": {"model": []}}):
        assert valid_recovery(bad) is None


@pytest.mark.parametrize("selected", ["primary", "fallback"])
@pytest.mark.parametrize("route_kind", ["foreign", "custom", "healed", "unroutable", "malformed"])
def test_nested_routes_are_sanitized_before_credential_resolution(monkeypatch, selected, route_kind):
    """Real row -> deadline selection -> runtime binding, with inert credentials only."""
    from tui_gateway.fallback_recovery import build_override
    from hermes_cli import runtime_provider as rp

    def no_network(*args, **kwargs):
        raise AssertionError("Provider requests forbidden")
    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    custom_url = "https://proxy.invalid/anthropic"
    cfg = {"providers": {"private": {"base_url": custom_url, "api_key": "inert-custom"}}}
    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setattr("tui_gateway.fallback_recovery._pool_reset_blocks", lambda *a: False)
    monkeypatch.setattr("tui_gateway.fallback_recovery.time.time", lambda: 1000)
    state = _state()
    state["retry_at"] = 0 if selected == "primary" else 2000
    target = state[selected]
    target.update(provider="anthropic", base_url="https://api.openai.com/v1", api_mode="chat_completions")
    if route_kind == "custom":
        target.update(base_url=custom_url, api_mode="anthropic_messages")
    elif route_kind == "healed":
        target.update(provider="removed-name", base_url=custom_url, api_mode="anthropic_messages")
    elif route_kind == "unroutable":
        target.update(provider="removed-name", base_url="https://removed.invalid")
    elif route_kind == "malformed":
        target["provider"] = ["anthropic"]
    row = {"model": "manual", "model_config": {"model": "manual", "provider": "openai-codex",
                                                   "fallback_recovery": state}}
    overrides = server._stored_session_runtime_overrides(row)
    if route_kind in {"unroutable", "malformed"}:
        assert overrides["model_override"]["model"] == "manual"
        assert "_fallback_recovery" not in overrides["model_override"]
        return
    route, _ = build_override(overrides["model_override"])
    # Credential resolver sees a normalized provider, never the removed identity.
    expected_provider = "custom:private" if route_kind == "healed" else "anthropic"
    expected_url = custom_url if route_kind in {"custom", "healed"} else "https://api.anthropic.com"
    resolutions = []

    def resolve(**kwargs):
        resolutions.append(kwargs)
        assert kwargs["requested"] == expected_provider
        return {"provider": expected_provider, "base_url": expected_url,
                "api_mode": "anthropic_messages", "api_key": "inert-credential-for-" + expected_provider}

    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)
    model, runtime = server._resolve_agent_model_runtime(route, route.get("provider"))
    assert resolutions and model == target["model"]
    # These are the exact URL/key fields passed to the agent/client constructor.
    assert runtime["base_url"] == expected_url
    assert runtime["api_key"] == "inert-credential-for-" + expected_provider
    assert runtime["api_mode"] == "anthropic_messages"


@pytest.mark.parametrize("edit", [None, "default", "provider", "base_url", "api_mode",
                                  "remove_base_url", "remove_api_mode", "null_base_url", "null_api_mode",
                                  "empty_base_url", "empty_api_mode"])
def test_deferred_profile_recovery_rechecks_route_at_build(monkeypatch, tmp_path, edit):
    """A profile may change after resume's acknowledgment but before agent construction."""
    from pathlib import Path
    import hermes_yaml as yaml

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    cfg = {"model": {"default": "primary", "provider": "anthropic",
                     "base_url": "https://proxy.invalid/anthropic", "api_mode": "anthropic_messages"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    state = _state()
    state["primary"].update(base_url=cfg["model"]["base_url"], api_mode=cfg["model"]["api_mode"])
    state["profile_intent"] = {"model": "primary", **{k: cfg["model"][k] for k in ("provider", "base_url", "api_mode")}}
    row = {"model": "fallback", "model_config": {"follow_profile_config": True, "fallback_recovery": state}}
    with server._profile_build_scope(tmp_path):
        overrides = server._stored_session_runtime_overrides(row)
        assert overrides["model_override"]["_fallback_recovery"]["retry_at"] == state["retry_at"]
        if edit:
            if edit.startswith("remove_"):
                cfg["model"].pop(edit.removeprefix("remove_"))
            elif edit.startswith("null_"):
                cfg["model"][edit.removeprefix("null_")] = None
            elif edit.startswith("empty_"):
                cfg["model"][edit.removeprefix("empty_")] = ""
            else:
                cfg["model"][edit] = {"default": "new-primary", "provider": "openrouter",
                                      "base_url": "https://new-proxy.invalid/anthropic",
                                      "api_mode": "chat_completions"}[edit]
            (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
        record = {"follow_profile_config": True, "resume_runtime_overrides": overrides,
                  "model_override": None, "cwd": str(tmp_path), "source": "desktop"}
        kwargs = server._deferred_build_agent_kwargs(record, None)
    assert bool(kwargs.get("model_override")) is (edit is None)


@pytest.mark.parametrize("bad", [None, [], "legacy", {}, {"model": "primary"},
                                {"model": "primary", "provider": "anthropic", "base_url": None, "api_mode": ""},
                                {"model": "primary", "provider": "anthropic", "base_url": [], "api_mode": ""}])
def test_optional_profile_intent_is_complete_secret_free_and_not_a_wildcard(monkeypatch, bad):
    from tui_gateway.fallback_recovery import profile_intent, recovery_matches_profile

    cfg = {"model": {"default": "primary", "provider": "anthropic", "api_key": "secret"}}
    intent = profile_intent(cfg)
    assert intent == {"model": "primary", "provider": "anthropic", "base_url": "", "api_mode": ""}
    state = {**_state(), "profile_intent": {**intent, "api_key": "secret", "client": {"secret": True}}}
    state["primary"].update(base_url="https://provider-default.invalid", api_mode="anthropic_messages")
    clean = valid_recovery(state)
    assert clean["profile_intent"] == intent and "secret" not in json.dumps(clean)
    assert recovery_matches_profile(clean, ("primary", "anthropic"), cfg)
    state["profile_intent"] = bad
    clean = valid_recovery(state)
    assert "profile_intent" not in clean
    assert not recovery_matches_profile(clean, ("primary", "anthropic"), cfg)
    # Ordinary explicit-session route recovery remains usable; canonical chats
    # without trustworthy historical intent follow fresh profile configuration.
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    row = {"model": "fallback", "model_config": {"fallback_recovery": state}}
    assert "_fallback_recovery" in server._stored_session_runtime_overrides(row)["model_override"]
    row["model_config"]["follow_profile_config"] = True
    assert server._stored_session_runtime_overrides(row) == {}
