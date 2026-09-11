"""Delegation consumes immutable runtimes without losing route configuration."""

from types import SimpleNamespace

import pytest

from agent.runtime_bundle import ResolvedRuntime
from tools.delegate_tool_config import _resolve_child_runtime, _resolve_delegation_credentials


def runtime(**updates):
    return ResolvedRuntime.from_mapping({
        "provider": "custom", "requested_provider": "custom:route",
        "model": "child-model", "api_mode": "chat_completions",
        "api_key": "test-key", "base_url": "https://route.invalid/v1",
        "extra_headers": {"X-Route": "route"}, "ssl_ca_cert": "route.pem",
        "default_query": {"route": "one"}, "timeout": 23,
        "request_overrides": {"extra_body": {"thinking": {"type": "disabled"}}},
        **updates,
    })


def parent(rt):
    return SimpleNamespace(
        model=rt.model, provider=rt.provider, api_mode=rt.api_mode,
        api_key=rt.api_key, base_url=rt.base_url, _resolved_runtime=rt,
        _client_kwargs={}, request_overrides=rt.get("request_overrides"),
    )


@pytest.mark.parametrize("direct", [False, True])
def test_frozen_provider_overrides_are_merged_and_isolated(monkeypatch, direct):
    rt = runtime()
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: rt)
    cfg = {"provider": "custom:route", "request_overrides": {"extra_body": {"route": ["child"]}}}
    if direct:
        cfg["base_url"] = rt.base_url
    creds = _resolve_delegation_credentials(cfg, parent(rt))
    overrides = creds["request_overrides"]
    assert overrides["extra_body"]["thinking"]["type"] == "disabled"
    overrides["extra_body"]["thinking"]["type"] = "enabled"
    overrides["extra_body"]["route"].append("changed")
    assert rt["request_overrides"]["extra_body"]["thinking"]["type"] == "disabled"
    assert cfg["request_overrides"]["extra_body"]["route"] == ["child"]


@pytest.mark.parametrize("cfg", [{"model": " auto "}, {"provider": "AUTO"}, {"model": "auto", "provider": "auto"}])
def test_auto_inherits_live_parent_without_resolving_again(monkeypatch, cfg):
    def unexpected(**kwargs):
        pytest.fail("auto must inherit the live parent, not resolve ambient config")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", unexpected)
    creds = _resolve_delegation_credentials(cfg, parent(runtime()))
    assert creds["model"] is None
    assert creds["provider"] is None


def test_child_derives_complete_runtime_and_keeps_parent_immutable():
    rt = runtime()
    child = _resolve_child_runtime(
        parent(rt), {}, rt.api_key, model="other-model", override_provider=None,
        override_base_url=None, override_api_key=None, override_api_mode=None,
        override_max_tokens=None, override_acp_command=None, override_acp_args=None,
    )
    inherited = child["resolved_runtime"]
    assert inherited.model == "other-model"
    assert inherited.extra_headers == rt.extra_headers
    assert inherited.ssl_ca_cert == rt.ssl_ca_cert
    assert inherited["default_query"] == rt["default_query"]
    assert inherited["timeout"] == rt["timeout"]
    assert rt.model == "child-model"
