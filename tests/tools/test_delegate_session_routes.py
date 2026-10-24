"""Anonymous retained routes refresh credentials without silently changing endpoints."""

import json
from types import SimpleNamespace

import pytest

from tools import delegate_session_routes as routes
from tools import delegate_tool_config as config


def _agent(endpoint="https://private.example/v1", key="initial-key", provider="custom"):
    return SimpleNamespace(base_url=endpoint, api_key=key, provider=provider, requested_provider=provider,
                           model="selected-model", api_mode="chat_completions")


def _record(descriptor, provider="custom"):
    return {"provider": provider, "model": "selected-model", "route_descriptor": descriptor}


def test_direct_route_uses_current_credentials_and_refuses_changed_endpoint(monkeypatch):
    selected = {"base_url": "https://private.example/v1", "api_key": "initial-key", "model": "selected-model"}
    cfg = selected
    monkeypatch.setattr(config, "_load_config", lambda: cfg)
    parent = _agent("https://parent.example/v1", provider="openai")
    child = _agent()
    descriptor = routes.capture_route(child, routing_cfg=selected)
    encoded = json.dumps(descriptor)
    assert "initial-key" not in encoded and "private.example" not in encoded
    selected["api_key"] = "rotated-key"
    resumed = routes.resume_route(parent, _record(descriptor), "another-model")
    resolved = config._resolve_delegation_credentials(resumed, parent)
    assert resolved["api_key"] == "rotated-key"
    assert resolved["model"] == "another-model"
    assert resolved["base_url"] == selected["base_url"]
    selected["base_url"] = "https://different.example/v1"
    with pytest.raises(ValueError, match="no longer matches"):
        routes.resume_route(parent, _record(descriptor))


def test_parent_route_does_not_adopt_new_delegation_default(monkeypatch):
    cfg = {}
    monkeypatch.setattr(config, "_load_config", lambda: cfg)
    parent = _agent()
    descriptor = routes.capture_route(_agent(), routing_cfg=cfg)
    cfg.update(base_url="https://new-default.example/v1", api_key="wrong-key")
    parent.api_key = "rotated-parent-key"
    resumed = routes.resume_route(parent, _record(descriptor))
    assert resumed == {"model": "selected-model"}
    parent.provider = parent.requested_provider = "openai"
    with pytest.raises(ValueError, match="provider has changed"):
        routes.resume_route(parent, _record(descriptor))
    parent.provider = parent.requested_provider = "custom"
    parent.base_url = "https://another-parent.example/v1"
    with pytest.raises(ValueError, match="no longer matches"):
        routes.resume_route(parent, _record(descriptor))


def test_named_provider_keeps_its_identity_when_defaults_change(monkeypatch):
    cfg = {"provider": "chosen-profile", "model": "selected-model"}
    monkeypatch.setattr(config, "_load_config", lambda: cfg)
    descriptor = routes.capture_route(_agent(provider="chosen-profile"), routing_cfg=cfg)
    cfg.update(provider="different-profile", base_url="https://wrong.example", api_key="wrong-key")
    resumed = routes.resume_route(_agent(), _record(descriptor, "chosen-profile"), "new-model")
    assert resumed["provider"] == "chosen-profile"
    assert resumed["model"] == "new-model"
    assert "base_url" not in resumed and "api_key" not in resumed


def test_external_route_captures_actual_endpoint_and_refuses_unrelated_default(monkeypatch):
    cfg = {"provider": "different-profile", "model": "some-model"}
    monkeypatch.setattr(config, "_load_config", lambda: cfg)
    actual = {"base_url": "https://private.example/v1", "api_key": "aux-key"}
    descriptor = routes.capture_route(_agent(), routing_cfg=actual)
    assert descriptor["kind"] == "default"
    cfg.clear()
    cfg.update(base_url="https://wrong.example/v1", api_key="wrong-key")
    with pytest.raises(ValueError, match="no longer matches"):
        routes.resume_route(_agent(), _record(descriptor))


def test_endpoint_identity_normalizes_equivalent_urls_but_keeps_route_path():
    assert routes._endpoint_fingerprint("https://USER:SECRET@EXAMPLE.com:443/v1/") == routes._endpoint_fingerprint("https://example.com/v1")
    assert routes._endpoint_fingerprint("https://example.com/v1") != routes._endpoint_fingerprint("https://example.com/v2")
