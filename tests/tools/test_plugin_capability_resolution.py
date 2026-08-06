"""Minimal core seam for external delegation capability policy."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def test_external_capability_hook_returns_child_controls(monkeypatch):
    from tools.delegate_tool import _resolve_plugin_capability

    expected = {
        "provider": "xai",
        "model": "xai/grok-4.5",
        "context": "specialist instructions",
        "toolsets": ["web"],
        "fallback_models": [],
        "workload": "research",
    }
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [expected] if name == "resolve_delegation_capability" else [],
    )
    assert _resolve_plugin_capability("researcher", "leaf") == expected


def test_external_capability_hook_fails_closed_when_unresolved(monkeypatch):
    from tools.delegate_tool import _resolve_plugin_capability

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])
    with pytest.raises(ValueError, match="not resolved"):
        _resolve_plugin_capability("unknown", "leaf")


def test_external_capability_hook_rejects_ambiguous_plugins(monkeypatch):
    from tools.delegate_tool import _resolve_plugin_capability

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [{"model": "a/b"}, {"model": "c/d"}],
    )
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_plugin_capability("coder", "leaf")


@pytest.mark.parametrize(
    "bad",
    [
        {"provider": "xai", "model": "broken", "context": "x", "fallback_models": []},
        {"provider": "xai", "model": "xai/grok", "context": 3, "fallback_models": []},
        {"provider": "xai", "model": "xai/grok", "context": "x", "fallback_models": ["broken"]},
        {"provider": "xai", "model": "xai/grok", "context": "x", "fallback_models": [], "toolsets": "web"},
    ],
)
def test_external_capability_hook_rejects_malformed_controls(monkeypatch, bad):
    from tools.delegate_tool import _resolve_plugin_capability

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [bad])
    with pytest.raises(ValueError, match="invalid"):
        _resolve_plugin_capability("coder", "leaf")


def test_capability_fallback_overrides_parent_chain():
    from tools.delegate_tool import _build_child_agent

    parent = MagicMock()
    parent._fallback_chain = [{"provider": "anthropic", "model": "wrong"}]
    parent.enabled_toolsets = ["web"]
    parent.valid_tool_names = []
    parent._delegate_depth = 0
    with patch("run_agent.AIAgent") as agent:
        agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="policy fallback",
            context=None,
            toolsets=None,
            model="openai-codex/gpt-5.6-sol",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            override_fallback_model=[{"provider": "xai", "model": "grok-4.5"}],
        )
    assert agent.call_args.kwargs["fallback_model"] == [
        {"provider": "xai", "model": "grok-4.5"}
    ]


def test_capability_fallback_credentials_are_preflighted(monkeypatch):
    from tools.delegate_tool import _resolve_capability_fallbacks

    seen = []

    def resolve(config, parent):
        seen.append((config["provider"], config["model"]))
        return {
            "provider": config["provider"],
            "model": config["model"],
            "base_url": "https://example.invalid",
            "api_key": "secret",
            "api_mode": "chat_completions",
            "request_overrides": None,
        }

    monkeypatch.setattr("tools.delegate_tool._resolve_delegation_credentials", resolve)
    result = _resolve_capability_fallbacks(
        ["xai-oauth/grok-4.5"], {"provider": "unused"}, object()
    )
    assert seen == [("xai-oauth", "grok-4.5")]
    assert result == [{
        "provider": "xai-oauth",
        "model": "grok-4.5",
        "base_url": "https://example.invalid",
        "api_key": "secret",
        "api_mode": "chat_completions",
        "request_overrides": None,
    }]


def test_capability_primary_credentials_use_bare_matching_model_ref():
    from tools.delegate_tool import _bare_capability_model

    assert _bare_capability_model(
        "xai-oauth/grok-4.5", "xai-oauth"
    ) == "grok-4.5"
    assert _bare_capability_model("grok-4.5", "xai-oauth") == "grok-4.5"


def test_trusted_coder_capability_can_elevate_child_toolsets():
    from tools.delegate_tool import _build_child_agent

    parent = MagicMock()
    parent._fallback_chain = []
    parent.enabled_toolsets = ["delegation", "skills", "memory"]
    parent.valid_tool_names = []
    parent._delegate_depth = 0
    with patch("run_agent.AIAgent") as agent:
        agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="write code",
            context=None,
            toolsets=["terminal", "file", "codex", "gitnexus"],
            model="openai-codex/gpt-5.6-sol",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            trusted_toolset_elevation=True,
        )
    enabled = set(agent.call_args.kwargs["enabled_toolsets"])
    assert {"terminal", "file", "codex", "gitnexus"} <= enabled


def test_untrusted_specialist_cannot_elevate_child_toolsets():
    from tools.delegate_tool import _build_child_agent

    parent = MagicMock()
    parent._fallback_chain = []
    parent.enabled_toolsets = ["delegation", "web"]
    parent.valid_tool_names = []
    parent._delegate_depth = 0
    with patch("run_agent.AIAgent") as agent:
        agent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="research",
            context=None,
            toolsets=["web", "terminal", "file"],
            model="xai/grok-4.5",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            trusted_toolset_elevation=False,
        )
    enabled = set(agent.call_args.kwargs["enabled_toolsets"])
    assert enabled == {"web"}
