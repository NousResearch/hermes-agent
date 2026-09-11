"""Real config -> child constructor -> SDK wire, with HTTP intercepted locally."""

import json
from types import SimpleNamespace

import certifi
import httpx
import pytest

from agent.runtime_bundle import ResolvedRuntime
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_config import _resolve_delegation_credentials


@pytest.fixture
def wire_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "0")
    config = {
        "model": {"default": "ambient-model", "provider": "custom", "context_length": 65536},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 3},
        "custom_providers": [{
            "name": "child-route", "base_url": "https://child.invalid/v1",
            "api_key": "child-key", "model": "gpt-4o-mini",
            "extra_headers": {"X-Route": "child"}, "ssl_ca_cert": certifi.where(),
        }],
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    requests, clients, verification = [], [], []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json={
                "id": "msg-test", "type": "message", "role": "assistant", "model": "claude-test",
                "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })
        return httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        })

    def http_client(base_url="", *, verify=True, **kwargs):
        verification.append(verify)
        client = httpx.Client(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(http_client))
    monkeypatch.setattr("agent.process_bootstrap.build_keepalive_http_client", http_client)
    # Metadata discovery is unrelated to routing and must not hit a provider.
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 65536)
    yield requests, verification
    for client in clients:
        client.close()


def make_parent(api_mode):
    rt = ResolvedRuntime.from_mapping({
        "provider": "custom", "requested_provider": "parent-route", "model": "gpt-4o-mini",
        "api_mode": api_mode, "api_key": "parent-key", "base_url": "https://parent.invalid/v1",
        "extra_headers": {"X-Route": "parent"}, "ssl_ca_cert": certifi.where(),
        "default_query": {"route": "parent"}, "timeout": 23,
        "request_overrides": {"extra_body": {"thinking": {"type": "disabled"}}},
    })
    return AIAgent(
        resolved_runtime=rt, enabled_toolsets=["file"], quiet_mode=True,
        skip_memory=True, skip_context_files=True, skip_background_review=True,
    )


def child_of(parent, creds=None):
    creds = creds or {}
    return _build_child_agent(
        0, "test", None, None, creds.get("model"), 2, 1, parent,
        override_provider=creds.get("provider"), override_base_url=creds.get("base_url"),
        override_api_key=creds.get("api_key"), override_api_mode=creds.get("api_mode"),
        override_request_overrides=creds.get("request_overrides"),
        override_runtime=creds.get("resolved_runtime"),
    )


def send(agent):
    if agent.api_mode == "anthropic_messages":
        client = agent._create_request_anthropic_client(reason="test")
        result = client.messages.create(model=agent.model, max_tokens=8, messages=[{"role": "user", "content": "test"}])
        agent._close_request_anthropic_client(client, reason="request_complete")
        assert result.content[0].text == "OK"
    else:
        client = agent._create_request_openai_client(reason="test")
        result = client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "test"}])
        agent._close_request_openai_client(client, reason="request_complete")
        assert result.choices[0].message.content == "OK"


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_nested_children_keep_runtime_and_own_clients(wire_env, api_mode):
    requests, verification = wire_env
    parent = make_parent(api_mode)
    child = child_of(parent, _resolve_delegation_credentials({"model": "auto", "provider": "auto"}, parent))
    grandchild = child_of(child)
    try:
        child_client = child._anthropic_client if api_mode == "anthropic_messages" else child.client
        parent_client = parent._anthropic_client if api_mode == "anthropic_messages" else parent.client
        assert child_client is not parent_client
        child.request_overrides["extra_body"]["thinking"]["type"] = "enabled"
        assert parent.request_overrides["extra_body"]["thinking"]["type"] == "disabled"
        assert grandchild.request_overrides["extra_body"]["thinking"]["type"] == "disabled"
        grandchild._swap_credential(SimpleNamespace(id="rotated", runtime_api_key="rotated-key", runtime_base_url=grandchild.base_url))
        send(grandchild)
        assert requests[-1].url.host == "parent.invalid"
        assert requests[-1].url.params["route"] == "parent"
        assert requests[-1].headers["X-Route"] == "parent"
        assert "rotated-key" in (requests[-1].headers.get("Authorization", "") + requests[-1].headers.get("X-Api-Key", ""))
        assert grandchild._resolved_runtime.api_key == "rotated-key"
        assert parent.api_key == child.api_key == "parent-key"
        child_client.close()
        send(parent)
        assert requests[-1].headers["X-Route"] == "parent"
        assert verification and all(value is not False for value in verification)
    finally:
        grandchild.close()
        child.close()
        parent.close()


@pytest.mark.parametrize("direct", [False, True])
def test_named_provider_resolves_in_profile_and_does_not_inherit_parent_route(wire_env, direct):
    requests, _ = wire_env
    parent = make_parent("chat_completions")
    cfg = {"provider": "child-route", "model": "auto"}
    if direct:
        cfg.update(base_url="https://child.invalid/v1", api_key="child-key")
    creds = _resolve_delegation_credentials(cfg, parent)
    child = child_of(parent, creds)
    try:
        send(child)
        assert requests[-1].url.host == "child.invalid"
        assert requests[-1].headers["Authorization"] == "Bearer child-key"
        assert requests[-1].headers["X-Route"] == "child"
        assert "route" not in requests[-1].url.params
        assert not child._fallback_chain
        assert child.capabilities == {}
        assert child._resolved_runtime.ssl_ca_cert == certifi.where()
        assert parent._resolved_runtime.extra_headers["X-Route"] == "parent"
    finally:
        child.close()
        parent.close()


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_fallback_resolver_request_rebuild_and_child_share_route_not_clients(wire_env, api_mode):
    from agent.chat_completion_helpers import try_activate_fallback
    requests, _ = wire_env
    parent = make_parent("chat_completions")
    parent._fallback_chain = [{"provider": "child-route", "model": "gpt-4o-mini", "api_mode": api_mode}]
    child = None
    try:
        assert try_activate_fallback(parent)
        assert parent._resolved_runtime.ssl_ca_cert == certifi.where()
        send(parent)
        assert requests[-1].url.host == "child.invalid"
        assert requests[-1].headers["X-Route"] == "child"
        assert "route" not in requests[-1].url.params
        child = child_of(parent)
        send(child)
        assert requests[-1].url.host == "child.invalid"
        assert requests[-1].headers["X-Route"] == "child"
        assert child._resolved_runtime.ssl_ca_cert == parent._resolved_runtime.ssl_ca_cert
    finally:
        if child is not None:
            child.close()
        parent.close()
