"""Delegated direct endpoints may inherit credentials only within their authority."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_config import (
    _resolve_child_credential_pool,
    _resolve_delegation_credentials,
)


def _parent(**overrides):
    values = {
        "base_url": "https://parent.example/v1",
        "api_key": "surface-parent-key",
        "provider": "custom",
        "requested_provider": "custom:parent",
        "api_mode": "chat_completions",
        "model": "parent-model",
        "client": None,
        "_client_kwargs": {
            "base_url": "https://parent.example/v1",
            "api_key": "live-parent-key",
        },
        "_credential_pool": None,
        "_fallback_chain": None,
        "_session_db": None,
        "_delegate_depth": 0,
        "_active_children": [],
        "_active_children_lock": None,
        "_print_fn": None,
        "tool_progress_callback": None,
        "thinking_callback": None,
        "enabled_toolsets": [],
        "disabled_toolsets": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_direct_endpoint_accepts_canonical_same_authority_without_explicit_key():
    parent = _parent(
        base_url="https://stale.example/v1",
        _client_kwargs={
            "base_url": "https://EXAMPLE.com./v1",
            "api_key": "live-parent-key",
        },
    )

    result = _resolve_delegation_credentials(
        {"base_url": "https://example.com:443/other"},
        parent,
    )

    assert result["api_key"] is None


def test_direct_endpoint_rejects_cross_authority_without_echoing_url_secrets():
    secret_url = "https://user:password@other.example/v1?token=query-secret"

    with pytest.raises(ValueError, match="authority mismatch") as error:
        _resolve_delegation_credentials({"base_url": secret_url}, _parent())

    assert "password" not in str(error.value)
    assert "query-secret" not in str(error.value)


def test_cross_authority_fails_before_child_agent_construction():
    parent = _parent()

    with patch("run_agent.AIAgent") as agent_cls:
        with pytest.raises(ValueError, match="authority mismatch"):
            _build_child_agent(
                task_index=0,
                goal="cross authority",
                context=None,
                toolsets=None,
                model="child-model",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider="custom",
                override_base_url="https://other.example/v1",
                override_api_key=None,
            )

    agent_cls.assert_not_called()


def test_same_live_authority_inherits_live_key_instead_of_stale_surface_key():
    parent = _parent(
        base_url="https://stale.example/v1",
        api_key="stale-surface-key",
        _client_kwargs={
            "base_url": "http://127.0.0.1:20128/v1/",
            "api_key": "live-hosted-key",
        },
    )

    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="same authority",
            context=None,
            toolsets=None,
            model="hosted-model",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            override_provider="custom",
            override_base_url="http://127.0.0.1:20128/worker",
            override_api_key=None,
            runtime_cfg={"provider": "custom"},
        )

    assert agent_cls.call_args.kwargs["api_key"] == "live-hosted-key"


def test_cross_authority_with_explicit_child_key_succeeds():
    result = _resolve_delegation_credentials(
        {"base_url": "https://other.example/v1", "api_key": "explicit-child-key"},
        _parent(),
    )

    assert result["api_key"] == "explicit-child-key"


def test_different_named_custom_provider_cannot_borrow_key_on_shared_gateway():
    parent = _parent(
        base_url="https://gateway.example/v1",
        requested_provider="custom:alpha",
        _client_kwargs={
            "base_url": "https://gateway.example/v1",
            "api_key": "alpha-key",
        },
    )

    with pytest.raises(ValueError, match="credential identity mismatch"):
        _resolve_delegation_credentials(
            {
                "provider": "custom:beta",
                "base_url": "https://gateway.example/v1",
            },
            parent,
        )


def test_direct_child_construction_cannot_borrow_across_named_custom_identities():
    parent = _parent(
        base_url="https://gateway.example/v1",
        requested_provider="custom:alpha",
        _client_kwargs={
            "base_url": "https://gateway.example/v1",
            "api_key": "alpha-key",
        },
    )

    with patch("run_agent.AIAgent") as agent_cls:
        with pytest.raises(ValueError, match="credential identity mismatch"):
            _build_child_agent(
                task_index=0,
                goal="direct named override",
                context=None,
                toolsets=None,
                model="beta-model",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider="custom:beta",
                override_base_url="https://gateway.example/worker",
                override_api_key=None,
                runtime_cfg={"provider": "custom:alpha"},
            )

    agent_cls.assert_not_called()


def test_provider_only_direct_child_cannot_borrow_named_custom_parent_key():
    parent = _parent(
        requested_provider="custom:alpha",
        _client_kwargs={
            "base_url": "https://parent.example/v1",
            "api_key": "alpha-key",
        },
    )

    with patch("run_agent.AIAgent") as agent_cls:
        with pytest.raises(ValueError, match="credential identity mismatch"):
            _build_child_agent(
                task_index=0,
                goal="provider-only named override",
                context=None,
                toolsets=None,
                model="beta-model",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider="custom:beta",
                override_base_url=None,
                override_api_key=None,
                runtime_cfg={},
            )

    agent_cls.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "minimax"])
def test_provider_only_direct_child_cannot_borrow_parent_key_for_different_provider(provider):
    parent = _parent(
        _client_kwargs={
            "base_url": "https://parent.example/v1",
            "api_key": "alpha-key",
        },
    )

    with patch("run_agent.AIAgent") as agent_cls:
        with pytest.raises(ValueError, match="provider credential mismatch"):
            _build_child_agent(
                task_index=0,
                goal="provider-only override",
                context=None,
                toolsets=None,
                model="child-model",
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                override_provider=provider,
                override_base_url=None,
                override_api_key=None,
                runtime_cfg={},
            )

    agent_cls.assert_not_called()


def test_different_named_custom_provider_uses_its_own_pool_on_shared_gateway(monkeypatch):
    parent_pool = object()
    beta_pool = object()
    parent = _parent(_credential_pool=parent_pool)

    monkeypatch.setattr(
        "agent.credential_pool.get_custom_provider_pool_key",
        lambda _url, provider_name=None: provider_name,
    )
    monkeypatch.setattr(
        "tools.delegate_tool_config._loaded_pool",
        lambda key: beta_pool if key == "custom:beta" else None,
    )

    resolved = _resolve_child_credential_pool(
        "custom",
        parent,
        "https://parent.example/v1",
        effective_requested_provider="custom:beta",
    )

    assert resolved is beta_pool
    assert resolved is not parent_pool


def test_native_sdk_route_builds_with_its_own_auth_sentinel():
    parent = _parent(provider="openrouter", requested_provider="openrouter")
    cfg = {"provider": "bedrock", "model": "amazon.nova-pro-v1:0"}
    native_runtime = {
        "provider": "bedrock",
        "model": "amazon.nova-pro-v1:0",
        "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "api_key": "aws-sdk",
        "api_mode": "bedrock_converse",
    }

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=native_runtime,
    ):
        credentials = _resolve_delegation_credentials(cfg, parent)

    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="native SDK",
            context=None,
            toolsets=None,
            model=credentials["model"],
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            override_provider=credentials["provider"],
            override_base_url=credentials["base_url"],
            override_api_key=credentials["api_key"],
            override_api_mode=credentials["api_mode"],
            runtime_cfg=cfg,
        )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["provider"] == "bedrock"
    assert kwargs["base_url"] == native_runtime["base_url"]
    assert kwargs["api_key"] == "aws-sdk"
    assert kwargs["api_mode"] == "bedrock_converse"


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "override_provider": "custom",
            "override_base_url": None,
            "override_api_key": None,
        },
        {
            "override_provider": "bedrock",
            "override_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "override_api_key": None,
        },
        {
            "override_provider": "external-process",
            "override_base_url": "https://external.invalid/v1",
            "override_api_key": None,
            "override_acp_command": "/bin/true",
            "override_acp_args": [],
        },
    ],
)
def test_provider_bundle_guard_preserves_same_provider_native_and_acp_transports(overrides):
    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="non-bearer transport",
            context=None,
            toolsets=None,
            model="transport-model",
            max_iterations=10,
            task_count=1,
            parent_agent=_parent(),
            runtime_cfg={},
            **overrides,
        )

    agent_cls.assert_called_once()


def test_live_empty_key_never_falls_back_to_stale_surface_credential():
    from tools.delegate_tool_config import _inherit_parent_endpoint
    parent = SimpleNamespace(
        _client_kwargs={"base_url": "https://live.invalid/v1", "api_key": None},
        client=None,
    )
    assert _inherit_parent_endpoint(parent, "https://stale.invalid/v1", "stale-key") == (
        "https://live.invalid/v1", None,
    )


@pytest.mark.parametrize("live_key", [None, ""])
def test_keyless_live_route_survives_actual_child_client_initialization(live_key):
    from run_agent import AIAgent
    endpoint = "http://127.0.0.1:9292/v1"
    parent = _parent(
        base_url="https://stale.invalid/v1", api_key="stale-key",
        _client_kwargs={"base_url": endpoint, "api_key": live_key},
    )
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=AssertionError("route was re-resolved")) as router,
        patch.object(AIAgent, "_create_openai_client", return_value=MagicMock()) as create,
    ):
        child = _build_child_agent(
            task_index=0, goal="keyless local task", context=None, toolsets=None,
            model="local-test-model", max_iterations=1, task_count=1,
            parent_agent=parent, override_provider="custom", override_base_url=endpoint,
        )
    router.assert_not_called()
    assert child.base_url == endpoint
    assert child.api_key == "no-key-required"
    assert create.call_args.args[0]["base_url"] == endpoint
    assert create.call_args.args[0]["api_key"] == "no-key-required"
    child.close()
