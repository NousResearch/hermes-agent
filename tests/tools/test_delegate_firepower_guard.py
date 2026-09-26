"""delegate_task per-call model override and firepower policy tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import run_agent

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


def _parent():
    return SimpleNamespace(_delegate_depth=0)


def test_schema_advertises_model_override_and_firepower_reason():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert props["model"]["type"] == "string"
    assert props["provider"]["type"] == "string"
    assert props["firepower_reason"]["type"] == "string"


def test_delegate_refuses_flagship_without_firepower_reason():
    result = cast(Any, delegate_task)(
        tasks=[{"goal": "Diagnose the concurrency failure"}],
        model="gpt-6-astra-900k",
        provider="openai-codex",
        parent_agent=_parent(),
    )
    assert "firepower_reason" in result


def test_justified_route_reaches_child_credential_config_and_audit_log():
    captured = {}

    def fake_resolve(cfg, _parent_agent):
        captured.update(cfg)
        return {
            "model": cfg.get("model"), "provider": cfg.get("provider"),
            "base_url": cfg.get("base_url"), "api_key": cfg.get("api_key"),
            "api_mode": cfg.get("api_mode"), "command": None, "args": None,
        }

    with (
        patch("tools.delegate_tool._load_config", return_value={
            "model": "claude-opus-5", "provider": "claude-apr",
            "base_url": "inherited-endpoint", "api_key": "inherited-key",
            "api_mode": "anthropic",
        }),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=fake_resolve),
        patch("tools.delegate_tool.logger.info") as audit_log,
    ):
        result = cast(Any, delegate_task)(
            model="gpt-6-astra-900k", provider="openai-codex",
            firepower_reason="cross-module concurrency diagnosis",
            parent_agent=_parent(),
        )

    assert "No tasks provided" in result
    assert captured["model"] == "gpt-6-astra-900k"
    assert captured["provider"] == "openai-codex"
    assert captured["base_url"] == ""
    assert captured["api_key"] == ""
    assert captured["api_mode"] == ""
    assert "cross-module concurrency diagnosis" in str(audit_log.call_args)


def test_live_dispatch_forwards_model_policy_fields():
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
        run_agent.AIAgent._dispatch_delegate_task(
            cast(Any, _parent()),
            {
                "tasks": [{"goal": "Implement the approved change"}],
                "model": "gpt-6-astra-900k", "provider": "openai-codex",
                "firepower_reason": "hard multi-file codegen",
            },
        )
    assert captured["model"] == "gpt-6-astra-900k"
    assert captured["provider"] == "openai-codex"
    assert captured["firepower_reason"] == "hard multi-file codegen"


def test_registry_fallback_forwards_model_policy_fields():
    from tools.registry import registry

    entry = registry.get_entry("delegate_task")
    assert entry is not None
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
        entry.handler({
            "tasks": [{"goal": "Implement the approved change"}],
            "model": "gpt-5.6-sol-900k", "provider": "openai-codex",
            "firepower_reason": None,
        }, parent_agent=_parent())
    assert captured["model"] == "gpt-5.6-sol-900k"
    assert captured["provider"] == "openai-codex"
    assert captured["firepower_reason"] is None
