"""Configured aliases are complete cron and delegation routes."""
from types import SimpleNamespace

import pytest


def _config():
    return {
        "model": {"default": "main", "provider": "other"},
        "model_aliases": {
            "lab-deep": {"model": "gpt-5.6-sol", "provider": "openai-codex",
                         "base_url": "https://chatgpt.com/backend-api/codex"},
            "private": {"model": "vendor/wire", "provider": "custom",
                        "base_url": "https://private.example/v1", "api_key": "alias-only-key"},
            "empty": {"model": "vendor/empty", "provider": "custom",
                      "base_url": "https://empty.example/v1"},
        },
    }


def test_cron_primary_alias_carries_one_complete_route(monkeypatch):
    from cron.scheduler import _CronJobConfig, _resolve_job_runtime
    from hermes_cli import runtime_provider
    calls = []
    def resolve(**kw):
        calls.append(kw)
        return {"model": kw["target_model"], "provider": kw["requested"],
                "base_url": kw.get("explicit_base_url"), "api_key": kw.get("explicit_api_key"),
                "api_mode": "chat_completions"}
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    cfg = _config()
    job = {"model": "private"}
    rt, model = _resolve_job_runtime(job, "test", _CronJobConfig(cfg, "private", cfg["model"], "other"))
    assert (model, rt["provider"], rt["base_url"], rt["api_key"], rt["api_mode"]) == (
        "vendor/wire", "custom", "https://private.example/v1", "alias-only-key", "chat_completions")
    assert calls[0]["requested"] == "custom"
    with pytest.raises(ValueError, match="conflicts with an explicit endpoint"):
        _resolve_job_runtime({**job, "base_url": "https://other.example/v1"}, "test",
                             _CronJobConfig(cfg, "private", cfg["model"], "other"))
    assert len(calls) == 1


def test_cron_applicable_fallback_alias_and_pin_isolation():
    from cron.scheduler import _cron_alias_fallbacks, _job_fallback_chain
    cfg = {**_config(), "fallback_providers": [{"model": "private", "provider": "custom"}]}
    chain = _cron_alias_fallbacks(_job_fallback_chain({}, cfg), cfg)
    assert chain[0]["model"] == "vendor/wire"
    assert chain[0]["base_url"] == "https://private.example/v1"
    assert chain[0]["api_key"] == "alias-only-key"
    assert not _job_fallback_chain({"model": "raw-special"}, cfg)
    assert not _cron_alias_fallbacks(_job_fallback_chain({"model": "raw-special"}, cfg), cfg)


def test_delegation_alias_does_not_borrow_parent_key(tmp_path, monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider
    from tools.delegate_tool_config import _resolve_delegation_credentials, _resolve_child_runtime
    cfg = _config()
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: cfg)
    def resolve(**kw):
        return {"model": kw["target_model"], "provider": kw["requested"],
                "base_url": kw.get("explicit_base_url"), "api_key": kw.get("explicit_api_key") or "",
                "api_mode": "codex_responses" if kw["requested"] == "openai-codex" else "chat_completions",
                "request_overrides": {"temperature": 0}, "capabilities": {"vision": True}}
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    parent = SimpleNamespace(model="old", provider="other", base_url="https://old.example/v1",
                             api_key="old-secret", api_mode="chat_completions", _fallback_chain=None,
                             request_overrides={}, capabilities={"vision": False})
    for alias, wire, provider, host, key, mode in (
        ("lab-deep", "gpt-5.6-sol", "openai-codex", "chatgpt.com", "no-key-required", "codex_responses"),
        ("private", "vendor/wire", "custom", "private.example", "alias-only-key", "chat_completions"),
        ("empty", "vendor/empty", "custom", "empty.example", "no-key-required", "chat_completions"),
    ):
        creds = _resolve_delegation_credentials({"model": alias}, parent)
        assert (creds["model"], creds["provider"], creds["api_mode"], creds["api_key"]) == (wire, provider, mode, key)
        child = _resolve_child_runtime(parent, {}, parent.api_key, model=creds["model"],
            override_provider=creds["provider"], override_base_url=creds["base_url"],
            override_api_key=creds["api_key"], override_api_mode=creds["api_mode"],
            override_acp_command=None, override_acp_args=None,
            override_capabilities=creds["capabilities"], override_alias_route=True)
        assert host in child["base_url"]
        assert child["api_key"] != "old-secret"
        assert child["capabilities"] == {"vision": True}
    with pytest.raises(ValueError, match="conflicts with an explicit endpoint"):
        _resolve_delegation_credentials({"model": "private", "base_url": "https://other.example/v1"}, parent)
